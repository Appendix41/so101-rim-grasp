"""Bounded, simulation-only SO-101 side-grasp experiment (metres, radians)."""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from contextlib import nullcontext

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--heights-mm", nargs="+", type=int, default=[60, 90, 120])
parser.add_argument("--grasp-mm", nargs="+", type=int, default=[30, 60, 100])
parser.add_argument("--diameter-mm", type=int, default=100)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--xy-jitter-mm", type=float, default=3.)
parser.add_argument("--approach", choices=("side", "rim"), default="side")
parser.add_argument("--retry-mm", type=int, help="One scripted retry at this absolute height after an upright failure")
parser.add_argument("--branch-probe", action="store_true",
                    help="Inject balanced lateral aiming errors and fork failed physical states (development only)")
parser.add_argument("--biases-mm", type=int, nargs="+", help="Cross these initial x errors at every start position")
parser.add_argument("--replay-vision", type=Path, help="Source branch run: retry each saved failure with fixed-left vs overhead vision")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (40 <= args.diameter_mm <= 110 and args.repeats > 0 and 0 <= args.xy_jitter_mm <= 25 and
        all(30 <= h <= 160 for h in args.heights_mm) and
        all(15 <= h <= 160 for h in args.grasp_mm)):
    parser.error("dimensions or repeats outside supported pilot range")
if len(set(args.heights_mm)) != len(args.heights_mm):
    parser.error("duplicate can heights")
if args.retry_mm is not None and not 15 <= args.retry_mm <= 160:
    parser.error("retry height out of range")
if args.replay_vision and (not args.branch_probe or not (args.replay_vision/"branch-audit.json").is_file()):
    parser.error("vision replay requires --branch-probe and an audited branch run")
if args.biases_mm and (not args.branch_probe or any(abs(b)>30 for b in args.biases_mm)):
    parser.error("biases require branch probe and magnitudes up to 30 mm")
if args.branch_probe and (args.retry_mm is not None or args.approach != "rim"
                          or min(args.grasp_mm)<35 or not args.enable_cameras
                          or not set(args.heights_mm) <= {60, 90, 120}):
    parser.error("branch probe requires rim, grasp height >=35 mm, cameras, development heights, no --retry-mm")
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(args)

import h5py
import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_ASSET_PATH, SO101_FOLLOWER_CFG
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES

DT = 1 / 120
MIN_LIFT = 0.02
MAX_TILT = 15.0
HOLD_S = 0.4
PHASES = ("settle", "align", "approach", "close", "lift", "hold")
DELTAS_MM = {"repeat": (0, 0), "repeat_control": (0, 0), "lower": (0, -20),
             "left": (-22, 0), "right": (22, 0)}


def physics_state(robot, can):
    arm, obj = robot.root_physx_view, can.root_physx_view
    p = obj.get_transforms().clone()
    return {"q": arm.get_dof_positions().clone(), "qd": arm.get_dof_velocities().clone(),
            "pose": p[:, [0, 1, 2, 6, 3, 4, 5]], "twist": obj.get_velocities().clone()}


def restore(robot, cans, active, snapshots, command):
    state = snapshots[active]
    robot.write_joint_state_to_sim(state["q"], state["qd"])
    robot.set_joint_position_target(command)
    robot.write_data_to_sim()
    for h, obj in cans.items():
        obj.write_root_link_pose_to_sim(snapshots[h]["pose"])
        obj.write_root_com_velocity_to_sim(snapshots[h]["twist"])
    actual = physics_state(robot, cans[active])
    for name, v in actual.items():
        ref = state[name]
        # Quaternions are equivalent up to sign.
        if name == "pose" and torch.sum(v[:, 3:] * ref[:, 3:]) < 0:
            v[:, 3:] *= -1
        torch.testing.assert_close(v, ref, rtol=0., atol=2e-6)
    return actual


def hollow_can(stage, path, h, r, park):
    """One dynamic body; local origin at the centre of the underside."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*park))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.08)
    wall, n = 0.002, 32

    def surface(shape):
        shape.CreateDisplayColorAttr([(0.024, 0.024, 0.024)])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        sim_utils.modify_collision_properties(
            shape.GetPath().pathString, sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
        )

    bottom = UsdGeom.Cylinder.Define(stage, path + "/Bottom")
    bottom.CreateHeightAttr(wall)
    bottom.CreateRadiusAttr(r)
    bottom.AddTranslateOp().Set((0., 0., wall / 2))
    surface(bottom)
    for i in range(n):
        a = 2 * math.pi * i / n
        piece = UsdGeom.Cube.Define(stage, f"{path}/Wall{i:02d}")
        piece.CreateSizeAttr(1)
        piece.AddTranslateOp().Set(((r-wall/2)*math.cos(a), (r-wall/2)*math.sin(a), h/2))
        piece.AddRotateZOp().Set(math.degrees(a))
        piece.AddScaleOp().Set((wall, 2*r*math.tan(math.pi/n), h))
        surface(piece)
    return RigidObject(RigidObjectCfg(prim_path=path, spawn=None))


class Planner:
    """Privileged diagnostic controller, not a policy. Read FK from the robot USD."""

    def __init__(self, r, approach):
        stage = Usd.Stage.Open(str(SO101_FOLLOWER_ASSET_PATH))
        root = str(stage.GetDefaultPrim().GetPath())
        self.links, limits = [], []
        for name in SINGLE_ARM_JOINT_NAMES[:-1]:
            j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{root}/joints/{name}"))
            q = j.GetLocalRot0Attr().Get()
            assert j.GetAxisAttr().Get() == "Z" and j.GetLocalPos1Attr().Get().GetLength() < 1e-6
            assert abs(j.GetLocalRot1Attr().Get().GetReal() - 1.) < 1e-6
            self.links.append((np.array(j.GetLocalPos0Attr().Get()),
                               Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()))
            limits.append((j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()))
        self.bounds = np.deg2rad(np.array(limits).T)
        self.tcp = np.array([r - 0.010, 0., -0.078])
        self.attitude = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        if approach == "rim":
            self.tcp = np.array([-.004, 0., -.094])
            self.attitude = np.eye(3)

    def fk(self, joints):
        pos, rot = np.zeros(3), np.eye(3)
        for (trans, pre), angle in zip(self.links, joints):
            pos = pos + rot @ trans
            rot = rot @ pre @ Rotation.from_rotvec([0., 0., angle]).as_matrix()
        return pos + rot @ self.tcp, rot

    def solve(self, xyz, initial):
        def residual(q):
            pos, rot = self.fk(q)
            return np.r_[pos - xyz, 0.07 * Rotation.from_matrix(self.attitude.T @ rot).as_rotvec()]
        return least_squares(residual, initial, bounds=self.bounds).x


def run():
    from rim_vision import centre_xy
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    table = sim_utils.CuboidCfg(size=(1.6, 1.6, .06), collision_props=sim_utils.CollisionPropertiesCfg())
    table.func("/World/Table", table, translation=(0., 0., -.03))
    robot_cfg = SO101_FOLLOWER_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos, robot_cfg.init_state.rot = (0., 0., 0.), (1., 0., 0., 0.)
    robot = Articulation(robot_cfg)
    cameras = {}
    if args.enable_cameras:
        light = sim_utils.DomeLightCfg(intensity=1700.)
        light.func("/World/Light", light)
        for name, path in [("overhead", "/World/Overhead"), ("wrist", "/World/Robot/gripper/Wrist")]:
            cfg = CameraCfg(prim_path=path, width=320, height=240, data_types=["rgb"],
                            spawn=sim_utils.PinholeCameraCfg(focal_length=22., horizontal_aperture=28.,
                                                            clipping_range=(.01, 5.)))
            if name == "wrist":
                cfg.offset = CameraCfg.OffsetCfg(pos=(-.001,.1,-.04),
                                                  rot=(-.404379,-.912179,-.0451242,.0486914), convention="ros")
            cameras[name] = Camera(cfg)
    r = args.diameter_mm / 2000
    parks = {h: (.6, -.60 + .13*i, .001) for i, h in enumerate(args.heights_mm)}
    cans = {h: hollow_can(sim.stage, f"/World/Can{h}", h/1000, r, parks[h]) for h in parks}
    sim.reset()
    ids, names = robot.find_joints(SINGLE_ARM_JOINT_NAMES, preserve_order=True)
    assert names == SINGLE_ARM_JOINT_NAMES
    planner = Planner(r, args.approach)
    rng = np.random.default_rng(args.seed)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=sim.device).reshape(1, -1)
    if cameras:
        cameras["overhead"].set_world_poses_from_view(tensor([.18,-.58,.52]), tensor([.025,-.21,.07]))
        for _ in range(5):
            sim.render()
    # Pair all can/grasp heights on exactly the same positions.
    xy_values = np.array([.060, -.265]) + rng.uniform(-1, 1, (args.repeats, 2)) * args.xy_jitter_mm/1000
    rows = []
    if args.replay_vision:
        source_path = args.replay_vision.resolve()
        records = {r["key"]: r for r in json.loads((source_path/"results.json").read_text())}
        families = json.loads((source_path/"branch-audit.json").read_text())["families"]
        parents = [r["parent"] for r in families]
        if set(args.heights_mm) != {records[p]["can_mm"] for p in parents} or args.repeats != 1+max(records[p]["repeat"] for p in parents):
            raise ValueError("Pass source heights and repeats to cover all saved failures")
    with (h5py.File(source_path/"attempts.h5") if args.replay_vision else nullcontext()) as src, h5py.File(args.out / "attempts.h5", "x") as f:
        if src is not None:
            assert src.attrs["diameter_mm"] == args.diameter_mm and src.attrs["approach"] == args.approach
            f.attrs["external_parent_run"] = str(source_path)
        f.attrs.update(dt=DT, joint_names=json.dumps(names), phases=json.dumps(PHASES), seed=args.seed,
                       diameter_mm=args.diameter_mm, hold_s=HOLD_S, min_lift_m=MIN_LIFT, max_tilt_deg=MAX_TILT,
                       xy_jitter_mm=args.xy_jitter_mm, approach=args.approach, branch_probe=args.branch_probe)
        for rep, xy in enumerate(xy_values):
            for h, can in cans.items():
                bias_mm = (22 if rep % 2 == 0 else -22) if args.branch_probe else 0
                attempts = [(z, None, "first", b) for z in args.grasp_mm for b in (args.biases_mm or [bias_mm])]
                if src is not None:
                    attempts = [(records[p]["grasp_mm"], p, name, records[p]["initial_bias_mm"])
                                for p in parents if (records[p]["repeat"], records[p]["can_mm"]) == (rep,h)
                                for name in ("left","vision")]
                for index, (z, parent, action_name, bias_mm) in enumerate(attempts):
                    first_aim = xy + [bias_mm/1000, 0.]
                    if src is not None:
                        p = src[parent]
                        first_aim = p.attrs["aim_xy_m"]
                        xy = p.attrs["start_xy_m"]
                        snapshot = {other: {"pose": tensor([*park,1.,0.,0.,0.]), "twist": tensor([0.]*6)}
                                    for other,park in parks.items()}
                        snapshot[h] = {k:tensor(v[:]) for k,v in p["privileged/snapshot"].items()}
                        saved_target = robot.data.default_joint_pos.clone()
                        saved_target[:,ids] = tensor(p["executed/joint_target"][-1])
                        parent_base = p["privileged/clearance_m_tilt_deg"][np.where(p["phase"][:]==0)[0][-1],0]
                    home = np.deg2rad([0., 0., 0., 0., 90., 95.])
                    if parent is None:
                        q_full = robot.data.default_joint_pos.clone()
                        q_full[:, ids] = tensor(home)
                        robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full))
                        robot.set_joint_position_target(q_full)
                        robot.reset()
                        for other_h, obj in cans.items():
                            init = (*xy, .001) if other_h == h else parks[other_h]
                            obj.write_root_pose_to_sim(tensor([*init, 1., 0., 0., 0.]))
                            obj.write_root_velocity_to_sim(tensor([0.] * 6))
                            obj.reset()
                        start = home
                        aim_xy = first_aim
                    else:
                        if args.branch_probe:
                            before_state = restore(robot, cans, h, snapshot, saved_target)
                        start = robot.data.joint_pos[0, ids].cpu().numpy().copy()
                        home[:5] = start[:5]
                        # Branch actions are relative to the old command, never to the true new can centre.
                        if action_name == "vision":
                            aim_xy = centre_xy(src[parent]["observed/overhead_rgb"][-1], r)
                        elif args.branch_probe:
                            aim_xy = first_aim + [DELTAS_MM[action_name][0]/1000, 0.]
                        else:
                            aim_xy = can.data.root_link_pose_w[0,:2].cpu().numpy().copy()
                    grasp = np.r_[aim_xy, z/1000]
                    if args.approach == "rim":
                        grasp[0] += r-.001
                    offset = [0., .110, 0.] if args.approach == "side" else [0., 0., .095]
                    close = -.12 if args.approach == "side" else -.17
                    pre = planner.solve(grasp + offset, home[:5])
                    grip = planner.solve(grasp, pre)
                    lift = planner.solve(grasp + [0., 0., .065], grip)
                    ik_err = np.linalg.norm(planner.fk(grip)[0] - grasp)
                    traj = ((home, 40), (np.r_[pre, home[-1]], 100), (np.r_[grip, home[-1]], 80),
                            (np.r_[grip, close], 60), (np.r_[lift, close], 80), (np.r_[lift, close], 72))
                    qlog, alog, poses, twists, signals, phases = [], [], [], [], [], []
                    images = {name: [] for name in cameras}
                    image_steps = []
                    if parent is not None:
                        before_q = start.copy()
                        before_pose = can.data.root_link_pose_w[0].cpu().numpy().copy()
                    for phase, (target, steps) in enumerate(traj):
                        for k in range(steps):
                            t = (k + 1) / steps
                            cmd = start + (target - start) * (t*t*(3-2*t))
                            robot.set_joint_position_target(tensor(cmd), joint_ids=ids)
                            robot.write_data_to_sim()
                            sim.step(render=False)
                            robot.update(DT)
                            can.update(DT)
                            pose = can.data.root_link_pose_w[0].cpu().numpy().copy()
                            twist = can.data.root_com_vel_w[0].cpu().numpy().copy()
                            up_z = np.clip(1 - 2*(pose[4]**2 + pose[5]**2), -1, 1)
                            tilt = math.acos(up_z)
                            # Lowest point of the entire cylinder, not its rising centre when it tips.
                            clearance = pose[2] + min(0., h/1000*up_z) - r*math.sin(tilt)
                            signals.append((clearance, math.degrees(tilt)))
                            poses.append(pose)
                            twists.append(twist)
                            alog.append(cmd)
                            qlog.append(robot.data.joint_pos[0, ids].cpu().numpy())
                            phases.append(phase)
                            if cameras and len(phases) % 8 == 0:
                                sim.render()
                                for name, cam in cameras.items():
                                    cam.update(8*DT, force_recompute=True)
                                    images[name].append(cam.data.output["rgb"][0,:,:,:3].cpu().numpy().copy())
                                image_steps.append(len(phases)-1)
                        start = target
                    signals = np.asarray(signals)
                    if not all(np.isfinite(v).all() for v in [signals, poses, twists, qlog]):
                        raise RuntimeError("non-finite simulation result")
                    hold = signals[-round(HOLD_S/DT):]
                    base = signals[np.flatnonzero(np.array(phases) == 0)[-1], 0] if parent is None else parent_base
                    lowest, tilt = float(hold[:,0].min()-base), float(hold[:,1].max())
                    peak = float(signals[np.array(phases) >= 4, 0].max()-base)
                    valid = abs(base) <= .003 and (parent is not None or signals[traj[0][1]-1, 1] < 3.)
                    label = ("invalid_reset" if not valid else "held" if lowest >= MIN_LIFT and tilt <= MAX_TILT
                             else "tipped" if tilt > MAX_TILT else "lost_lift" if peak >= MIN_LIFT else "not_lifted")
                    key = f"attempt_{len(rows):04d}"
                    row = dict(key=key, repeat=rep, can_mm=h, grasp_mm=z, label=label,
                               approach=args.approach,
                               approach_shift_mm=round(float(np.linalg.norm(poses[219][:2]-poses[39][:2]))*1000, 2),
                               min_hold_lift_mm=round(lowest*1000, 2), max_hold_tilt_deg=round(tilt, 2),
                               peak_lift_mm=round(peak*1000, 2), ik_error_mm=round(float(ik_err)*1000, 2))
                    group = f.create_group(key)
                    row.update(parent=parent or "", attempt=1 if parent is None else 2)
                    if args.branch_probe:
                        row.update(candidate=action_name, initial_bias_mm=bias_mm)
                    group.attrs.update(**row, start_xy_m=xy, aim_xy_m=aim_xy)
                    if args.branch_probe:
                        state = physics_state(robot, can) if parent is None else before_state
                        dest = "snapshot" if parent is None else "branch_start"
                        for name, value in state.items():
                            group.create_dataset(f"privileged/{dest}/{name}", data=value.cpu().numpy()[0])
                        if parent is not None and src is None:
                            group.create_dataset("candidate/delta_xz_mm", data=DELTAS_MM[action_name])
                        elif src is not None:
                            group.create_dataset("candidate/delta_xyz_mm", data=np.r_[1000*(aim_xy-first_aim),0.])
                    if parent is not None:
                        group.create_dataset("observed/before_retry_joint_pos", data=before_q)
                        group.create_dataset("privileged/before_retry_can_pose", data=before_pose)
                    for path, data in [("observed/joint_pos", qlog), ("executed/joint_target", alog),
                                       ("privileged/can_pose_wxyz", poses), ("privileged/can_com_twist", twists),
                                       ("privileged/clearance_m_tilt_deg", signals), ("phase", phases)]:
                        group.create_dataset(path, data=np.asarray(data), compression="gzip", shuffle=True)
                    if cameras:
                        group.create_dataset("observed/image_step", data=image_steps)
                        for name, frames in images.items():
                            group.create_dataset(f"observed/{name}_rgb", data=frames, chunks=(1,240,320,3),
                                                 compression="lzf")
                    f.flush()
                    rows.append(row)
                    print("ATTEMPT", json.dumps(row), flush=True)
                    if ((args.retry_mm is not None or args.branch_probe) and parent is None and label not in ("invalid_reset", "held")
                            and signals[-1,1] <= MAX_TILT and abs(signals[-1,0]-base) < .005):
                        parent_base = base
                        if args.branch_probe:
                            snapshot = {hh: physics_state(robot, obj) for hh, obj in cans.items()}
                            saved_target = robot.data.joint_pos_target.clone()
                            order = rng.permutation(list(DELTAS_MM)).tolist()
                            attempts[index+1:index+1] = [(z + DELTAS_MM[name][1], key, name, bias_mm) for name in order]
                        else:
                            attempts.insert(index + 1, (args.retry_mm, key, "retry", bias_mm))
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    # This headless worker writes HDF5 synchronously above. Kit teardown on this
    # host can spin indefinitely even after every writer has been closed.
    if args.headless:
        print(f"SWEEP_COMPLETE {len(rows)} attempts: {args.out}", flush=True)
        sys.stderr.flush()
        os._exit(0)
    sim.stop()


if __name__ == "__main__":
    try:
        run()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
