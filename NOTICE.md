# What's distributed

The MIT license covers the original code and documentation in this repository.
The author also permits reuse of the included simulation observations, replay videos,
audits and two task-specific nearest-neighbor model banks under the same terms.
Attribution: **Appendix41, SO-101 Rim Grasp**.

- The pretrained [LeRobot π0.5](https://huggingface.co/lerobot/pi05_base)
  and [PaliGemma tokenizer](https://huggingface.co/google/paligemma-3b-pt-224)
  are separate upstream materials. They, and π0.5-derived LoRA binaries, are
  **not** distributed here. Consult their model cards and access terms.
  The two `.npz` model banks here are fits to measured simulator outcomes,
  not exported foundation-model parameters.
- The simulation was made with [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim),
  [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and
  [leIsaac's SO-101](https://github.com/LightwheelAI/leisaac).
  Screenshots and videos show that simulated robot. No simulator, robot mesh,
  USD asset, vendor driver, paper figure or external demonstration is bundled.
- Data are simulated. No real camera captures, physical robot identifiers,
  host-specific calibration files, or sealed test trajectories are included.
