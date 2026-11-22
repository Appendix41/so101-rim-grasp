# Data card

All observations here were rendered by Isaac Sim. Images are unmarked RGB, 320 × 240. Joint positions are radians in the order shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. Load archives with `numpy.load(path, allow_pickle=False)`.

| File | Unit and fields | Use |
|---|---|---|
| [placement-train.npz](placement-train.npz) | 27 training layouts × 2 opposite color assignments × 3 decision frames = **162 paired-camera observations**. `overhead_rgb`, `wrist_rgb` (uint8, `[27,2,3,240,320,3]`), `joint_pos` (`[27,2,3,6]`), `uv` (`[27,2,2]`, one teacher target `(u,v)` per side), `seed`, `source`, `weight`. | First 19 layouts: visual-goal distillation. Last 8: learner-visited decision points for the DAgger round. The sampler mixes old and new; `weight` is not an action or a policy input. |
| [placement-dev-b.npz](placement-dev-b.npz) | 4 **different** layouts × 2 swaps = 8 starts. Three RGB/state observations per start; `pixels` are **logged** learner proposals. `eye`, `quat` (wxyz, camera to world), `K` are simulation camera calibrations. `menu` is the deliberately injected four-point menu in table XY metres. | Offline reproduction of selector decisions, **not training**. Natural alternatives are re-detected from RGB; injected alternatives come from the harness. Endpoints are stored separately in [dev-b.json](../evidence/dev-b.json). |

The archived planner used `12` for black far and `21` for black near; these IDs were not given to the learned policy. The same physical handoff can generate multiple comparison branches. Do not random-split frames, treat opposite colors as unrelated layouts, or count a reused physical suffix multiple times as an independent trial.

No sealed-test data, raw physical-camera data, raw episode HDF5 files or simulator asset bundles are included. The [export script](../tools/pack_observations.py) shows exactly which archive fields became public; it is not required to run the CPU examples. Each model bank has its own manifest in [models](../models).
