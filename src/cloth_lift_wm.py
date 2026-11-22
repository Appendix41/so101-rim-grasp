"""Nonparametric, skill-conditioned model of can clearance (mm) and tilt (deg)."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from cloth_task import LAYOUTS
from placement_scene import scene_split


ACTIONS = ("live", "vision")
TICKS = 832
HORIZON = TICKS / 120
GROUPS = (288, 288, 12, 12)
FLOORS = (.025, .025, .01, .06)


def features(prefix):
    # Only sensor readings available at or before closure. No scene/layout ID.
    groups = []
    for key in ("overhead_rgb", "wrist_rgb", "joint_pos", "joint_vel"):
        past = prefix["observed/" + key][-4]
        now = prefix["terminal/" + key][:]
        if key.endswith("_rgb"):
            def pool(img):
                return img.reshape(6, 40, 8, 40, 3).mean((1, 3)).ravel() / 255.
            past, now = pool(past), pool(now)
        groups.append(np.r_[now, now - past])
    assert tuple(map(len, groups)) == GROUPS
    x = np.concatenate(groups)
    assert np.isfinite(x).all()
    return x


def timed_outcome(trajectory, error=False):
    eligible = (trajectory[:, 0] >= 20.) & (trajectory[:, 1] <= 15.)
    edges = np.diff(np.r_[False, eligible, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    spans = starts[stops-starts >= 48]
    return (float(spans[0] + 48) / 120) if len(spans) and not error else None


def physical(g):
    arr = g["privileged/signals"][:, :2].astype(np.float64)
    arr[:, 0] *= 1000.
    assert np.isfinite(arr).all()
    return arr


def forecast(model, x, action):
    ids = np.flatnonzero(model["action"] == action)
    trajectories = model["trajectories"][ids]
    diff = ((x - model["X"][ids]) / model["scale"]) ** 2
    dist2 = np.mean([g.mean(axis=1) for g in np.split(diff, np.cumsum(GROUPS)[:-1], axis=1)], axis=0)
    nearest = np.argsort(dist2, kind="stable")[:4]
    mix = np.exp(-.5 * dist2[nearest] / max(dist2[nearest[-1]], 1.e-12))
    weights = np.full(len(ids), .1 / len(ids))
    weights[nearest] += .9 * mix / mix.sum()
    # Score members of the forecast distribution, not its often nonphysical mean.
    times = [timed_outcome(t, e) for t, e in zip(trajectories, model["errors"][ids])]
    costs = [HORIZON + 12 if t is None else t for t in times]
    return dict(mean=np.einsum("n,ntd->td", weights, trajectories),
                cost=float(weights @ costs + .35*action),
                p_held=float(weights @ [t is not None for t in times]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(directory):
    meta = json.loads((directory / "freeze.json").read_text())
    path = directory / "wm.npz"
    assert sha(path) == meta["model_sha256"] and sha(Path(__file__)) == meta["code_sha256"]
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def episodes(run, split):
    plan = json.loads((run/"protocol.json").read_text())
    assert plan["protocol"] == "cloth-wm-home-v1" and plan["split"] == split
    with h5py.File(run/"prelift.h5") as f:
        assert set(f) == set(map(str,plan["scenes"]))
        assert f.attrs["adapter_sha256"] == plan["adapter_sha256"]
        assert not f.attrs["hardware_accessed"] and not f.attrs["software_joint_clipping"]
        assert json.loads(f.attrs["policy_metadata"])["adapter_loaded"]
        for s in plan["scenes"]:
            assert scene_split(s) == split and set(f[str(s)]) == set(LAYOUTS)
            for layout in LAYOUTS:
                assert set(f[f"{s}/{layout}"]) == set(plan["conditions"])
                for cond in plan["conditions"]:
                    cell = f[f"{s}/{layout}/{cond}"]
                    assert cell.attrs["split"] == split
                    pre = cell["prefix"]
                    raw = np.concatenate([pre[f"policy/raw_{t}"][:] for t in range(0,len(pre["phase"]),50)])
                    cmd = pre["executed/joint_target"][:]
                    np.testing.assert_array_equal(cmd, raw[:len(cmd)])
                    np.testing.assert_array_equal(cmd, pre["executed/drive"][:])
                    q = np.r_[pre["observed/joint_pos"][:,5], pre["terminal/joint_pos"][5]]
                    lo, hi = plan["gate_frames"]
                    gate = next((i for i in range(lo,min(hi,len(cmd))+1) if q[:i].max()>=1.
                                 and max(q[i-1],q[i],cmd[i-1,-1])<=plan["jaw_gate_rad"]),-1)
                    assert gate == cell.attrs["fork_frame"]
                    if gate == -1:
                        assert len(cmd)==200 and "branches" not in cell
                        yield (s,layout,cond), cell, pre, None
                        continue
                    action = cell.attrs["selected"] if cond=="selector" else cond
                    assert set(cell["branches"])=={action} and len(cmd)==gate
                    branch = cell["branches/"+action]
                    assert branch.attrs["slot"]==0 and not branch.attrs["privileged_controller"]
                    cmd2 = branch["executed/joint_target"][:]
                    assert cmd2.shape == (104,6) and len(branch["privileged/signals"])==TICKS
                    np.testing.assert_array_equal(cmd2,branch["executed/drive"][:])
                    if action=="live":
                        pieces = [raw[gate:]] + [branch[f"policy/raw_{t}"][:] for t in range(len(raw),gate+104,50)]
                        np.testing.assert_array_equal(cmd2,np.concatenate(pieces)[:104])
                    yield (s,layout,cond), cell, branch, action


def fit(run, out):
    X, actions, y, errors, keys, no_gate = [], [], [], [], [], []
    for key, cell, branch, action in episodes(run, "train"):
        if action is None:
            no_gate.append(key)
            continue
        keys.append(key)
        X.append(features(cell["prefix"]))
        actions.append(ACTIONS.index(action))
        y.append(physical(branch))
        errors.append(bool(branch.attrs["planning_error"]))
    X = np.stack(X)
    assert min(actions.count(i) for i in (0,1)) >= 4
    scale = np.maximum(X.std(0), np.repeat(FLOORS,GROUPS))
    out.mkdir(exist_ok=False)
    path = out/"wm.npz"
    np.savez_compressed(path,X=X,action=actions,trajectories=y,errors=errors,scale=scale)
    info = dict(model_sha256=sha(path),code_sha256=sha(Path(__file__)),
                train_protocol_sha256=sha(run/"protocol.json"), keys=keys, no_gate=no_gate,
                train_held={a:sum(timed_outcome(t,e) is not None for t,e,i in zip(y,errors,actions)
                                  if ACTIONS[i]==a) for a in ACTIONS},groups=GROUPS)
    (out/"freeze.json").write_text(json.dumps(info,indent=2))
    print("Frozen",len(keys),"trajectories",info["train_held"],info["model_sha256"])


def evaluate(run, out):
    model = load_frozen(out)
    rows = []
    for (s,l,cond),cell,b,action in episodes(run,"validation"):
        y = physical(b)
        t = timed_outcome(y,bool(b.attrs.get("planning_error","")))
        row = dict(scene=s,layout=l,condition=cond,action=action,gate=int(cell.attrs["fork_frame"]),
                   held=t is not None,first_s=t,planning_error=str(b.attrs.get("planning_error","")))
        if action is not None:
            x = features(cell["prefix"])
            preds = [forecast(model,x,i) for i in (0,1)]
            i = ACTIONS.index(action)
            row.update(pred_cost=[v["cost"] for v in preds],pred_p_held=[v["p_held"] for v in preds],
                       expected_time_capped=t if t is not None else HORIZON,
                       forecast_mae=abs(preds[i]["mean"]-y).mean(0).tolist(),
                       prior_mae=abs(model["trajectories"][model["action"]==i].mean(0)-y).mean(0).tolist())
            if cond=="selector":
                assert i == int(np.argmin(row["pred_cost"]))
                np.testing.assert_allclose(row["pred_cost"],cell.attrs["predicted_costs"],rtol=1.e-10,atol=1.e-10)
                np.testing.assert_allclose(x,cell["decision_features"][:],rtol=1.e-10,atol=1.e-10)
            if action=="vision":
                vs = list(b["vision"].values()) if "vision" in b else []
                row.update(wrist_used=sum(bool(v.attrs.get("wrist_used",False)) for v in vs), aims=len(vs))
        rows.append(row)
    summary, skill_errors = {}, {}
    for cond in (*ACTIONS,"selector"):
        rr = [r for r in rows if r["condition"]==cond]
        tt = [r["first_s"] for r in rr if r["held"] and r["gate"]!=-1]
        tc = [r["expected_time_capped"] for r in rr if r["gate"]!=-1]
        summary[cond] = dict(starts=len(rr),gates=len(tc),held=sum(r["held"] for r in rr),
                             retries=sum(r["action"]=="vision" for r in rr),
                             mean_success_s=np.mean(tt).item() if tt else None,
                             mean_capped_s=np.mean(tc).item() if tc else None)
        if cond in ACTIONS and tc:
            skill_errors[cond] = {k: np.mean([r[k] for r in rr if r["gate"]!=-1],0).tolist()
                                  for k in ("forecast_mae","prior_mae")}
    with (run/"wm_audit.json").open("x") as f:
        json.dump(dict(model_sha256=sha(out/"wm.npz"),summary=summary,skill_errors=skill_errors,rows=rows),
                  f,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=summary,skill_errors=skill_errors),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("fit","evaluate"))
    p.add_argument("run",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    (fit if args.mode=="fit" else evaluate)(args.run,args.out)
