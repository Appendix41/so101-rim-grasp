# SO-101 Can-Rim Manipulation

This is a single-GPU Isaac Sim study of a thin-rim grasp and black-cloth placement task using task-specific π0.5 LoRA modules, imitation learning, and DAgger.

## Policy

The system switches between two task-specific adapters on a frozen π0.5 base:

- **Grasp LoRA `af182185`:** predicts joint-action chunks for approaching, closing, and lifting the can.
- **Placement LoRA `9549e9c8`:** predicts an overhead pixel goal for the black cloth.
- **Execution:** a fixed geometric IK controller carries, lowers, releases, and retreats. The placement adapter does not output joint trajectories.

DAgger labels states visited by the learner, then mixes successful expert corrections with verified replay. The grasp update contains 5 prior corrections, 17 new corrections, and 116 capability replay windows. It updates rank-8, alpha-16 LoRA parameters for 80 steps; failed expert branches are excluded as action targets. The placement update uses 19 earlier paired examples plus 8 new on-policy layout pairs, trained for 192 steps.

## Results

| Evaluation | Parent | Updated policy |
|---|---:|---:|
| Grasp end-hold, 8 new development conditions | 6/8 | **7/8** |
| Grasp end-hold, 8 old short-can diagnostics | 1/8 | **4/8** |
| Grasp end-hold, 24 previously seen conditions | 12/24 | **14/24** |
| Placement DAgger, 8 matched starts | 4/8 | **7/8** |

The grasp candidate has regressions as well as gains and is not claimed to dominate its parent. In a separate 16-condition full-task comparison with world models and scripted visual regrasp disabled, first sustained lift was 15/16 for both grasp adapters; can retention at the IK handoff increased from 3/16 to 9/16, and strict full-task completion from 1/16 to 2/16. Placement motion still used geometric IK.

## Replays

- [Full-task success](../media/replays/pick-place.gif) · [MP4](../media/replays/pick-place.mp4)
- [Full-task overhang failure](../media/replays/pick-place-overhang.gif) · [MP4](../media/replays/pick-place-overhang.mp4)
- [Grasp recovery pair](../media/replays/grasp-recovery.gif) · [MP4](../media/replays/grasp-recovery.mp4)
- [Grasp regression pair](../media/replays/grasp-regression.gif) · [MP4](../media/replays/grasp-regression.mp4)
- [Placement color correction](../media/replays/placement-color.gif) · [MP4](../media/replays/placement-color.mp4)
- [Placement edge check](../media/replays/placement-margin.gif) · [MP4](../media/replays/placement-margin.mp4)

All clips show simulation. The placement-only clips start from an expert-assisted visual-grasp state and compare separately replayed placement branches.

## Scope

The robot is controlled at 15 Hz with 120 Hz physics; 50 is the action-chunk horizon, not the control frequency. The reported 10.76 GiB peak is for the training process, not a combined simulator-and-training measurement. No physical-robot success has been recorded for this task. See [experiments and ablations](experiments.md), [reproduction limits](reproduce.md), and [hardware trial status](../real/README.md).
