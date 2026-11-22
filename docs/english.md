# SO-101 Rim Grasp

A single-GPU study of getting a low-cost arm to pinch the wall of an open can, lift it, and put the **whole base** on a black cloth instead of a red distractor. The instruction always targets black; the cloths swap sides. All task scores here are from **Isaac Sim development layouts**. [Physical robot trials](../real/README.md) are pending.

[Watch the 21-second paired replay](../media/student-vs-world-model.mp4). The student picked the right color but left 31.1 mm of the can over the boundary. An action-conditioned model evaluated an alternative before release; the same carry primitive then placed the can with 20.9 mm of clearance.

| What changed? | What happened? |
|---|---|
| Grasp recovery | Continuing the grasp LoRA yielded 1/8 sustained lifts. Always performing a dual-camera visual regrasp yielded 8/8; a learned selector also scored 8/8 and chose regrasp every time. |
| Decision-point DAgger | A second π0.5 LoRA predicts an overhead goal pixel. An RGB teacher labeled states visited by the learner. On eight matched starts, v1 scored 4/8 and the updated student 7/8. It scored 7/8 again in a second group. |
| Online placement checking | With the same grasp stack, independent student goals scored **14/16**, and adding the placement outcome model scored **16/16**. Eight layouts, each color-swapped, not 16 independent layouts. |
| Deliberate stress test | Given 32 injected starting proposals, direct execution scored 8/32, the learned selector 28/32, and the simple inset-boundary rule **32/32**. The learned model missed four near-side overhangs. |

## Two lightweight world models

Here, a *world model* is a **task-specific, action-conditioned outcome predictor**. The visual teacher labels what the arm *should* do. These models instead learn from what *happened* when a candidate was executed in simulation, and compare candidates at inference time.

- **[Grasp: continue or regrasp?](../src/cloth_lift_wm.py)** At jaw closure, the model reads overhead and wrist RGB, joint positions and velocities, and their recent changes. Using 32 grasp trajectories, it forecasts the can’s clearance above the table and tilt over the next 6.9 seconds for two skills: continue π0.5 or run the visual regrasp. The controller chooses a skill; the model does not generate the regrasp motion.
- **[Placement: will the whole base fit?](../src/cloth_landing_wm.py)** Before transport, the model reads overhead color features around each proposed point and the current joint positions. From 39 distinct simulated placements, it estimates where the can center will land (position and surface: black cloth, red cloth, or neither), the minimum whole-base margin on the black cloth over the final two seconds, and a strict-success score. A negative margin means overhang. The controller ranks the π0.5 point against visual alternatives or holds the can if none qualifies. The same IK routine carries, lowers, releases and retreats.

Both use weighted nearest neighbors, closer to a [skill effect model](https://arxiv.org/abs/2109.08771) than a general-purpose simulator. Neither generates video, rolls out arbitrary joint commands, or supplies reward updates to π0.5. A [separate online RL-token experiment](../evidence/rl-side-study.json) did not improve sustained grasping.

**Released:** [six simulation videos](../media), [train pairs and development replay inputs](../data/README.md), [per-trial audits](../evidence), [two fitted outcome models](../models), [goal-LoRA training](../training/train_goal.py), and a [simulation-only grasp baseline](../sim/grasp_sweep.py). `python tools/audit.py` reproduces the counts without dependencies; `python tools/replay_choices.py` reruns the selector using the original development images and logged student proposals. See [reproduction](reproduce.md) and [upstream materials / licensing](../NOTICE.md).

**Limits:** the simulated can has a 2 mm **rigid** wall. This tests contact and recovery, not deforming or crushing a real shell. The main evaluation does not establish arbitrary color-instruction following, robustness to new can heights, or real-world success.
