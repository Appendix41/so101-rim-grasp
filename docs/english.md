# SO-101 Rim Grasp

A single-GPU study of getting a low-cost arm to pinch the wall of an open can, lift it, and put the **whole base** on a black cloth instead of a red distractor. The cloths swap sides. All task scores here are from **Isaac Sim development layouts**. [Physical robot trials](../real/README.md) are pending.

[Watch the 21-second paired replay](../media/student-vs-world-model.mp4). The student picked the right color but left 31.1 mm of the can over the boundary. An action-conditioned model evaluated an alternative before release; the same carry primitive then placed the can with 20.9 mm of clearance.

| What changed? | What happened? |
|---|---|
| Grasp recovery | Continuing the grasp LoRA yielded 1/8 sustained lifts. Always performing a dual-camera visual regrasp yielded 8/8; a learned selector also scored 8/8 and chose regrasp every time. |
| Decision-point DAgger | A second π0.5 LoRA predicts an overhead goal pixel. An RGB teacher labeled states visited by the learner. On eight matched starts, v1 scored 4/8 and the updated student 7/8. It scored 7/8 again in a second group. |
| Online placement checking | With the same grasp stack, independent student goals scored **14/16**, and adding the placement outcome model scored **16/16**. Eight layouts, each color-swapped, not 16 independent layouts. |
| Deliberate stress test | Given 32 injected starting proposals, direct execution scored 8/32, the learned selector 28/32, and the simple inset-boundary rule **32/32**. The learned model missed four near-side overhangs. |

The task uses overhead and wrist RGB, joint state, and a fixed black-cloth instruction. Carry, lowering, release and retreat are fixed IK skills. The two “world models” are lightweight action-conditioned outcome predictors, not video generators. The main improvement is from imitation post-training and online selection, not reward-based updates to π0.5. A [separate online RL-token experiment](../evidence/rl-side-study.json) did not improve sustained grasping.

**Released:** [six simulation videos](../media), [train pairs and development replay inputs](../data/README.md), [per-trial audits](../evidence), [two fitted outcome models](../models), [goal-LoRA training](../training/train_goal.py), and a [simulation-only grasp baseline](../sim/grasp_sweep.py). `python tools/audit.py` reproduces the counts without dependencies; `python tools/replay_choices.py` reruns the selector using the original development images and logged student proposals. See [reproduction](reproduce.md) and [upstream materials / licensing](../NOTICE.md).

**Limits:** the simulated can has a 2 mm **rigid** wall. This tests contact and recovery, not deforming or crushing a real shell. The main evaluation does not establish arbitrary color-instruction following, robustness to new can heights, or real-world success.
