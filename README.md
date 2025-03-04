# Whenever, Wherever: Towards Orchestrating Crowd Simulations with Spatio-Temporal Spawn Dynamics


## Authors
Public Repository for our ICRA2025 accepted paper. Thomas Kreutz, Max Mühlhäuser, and Alejandro Sanchez Guinea

## Abstract
Realistic crowd simulations are essential for immersive virtual environments, relying on both individual behaviors (microscopic dynamics) and overall crowd patterns (macroscopic characteristics). While recent data-driven methods like deep reinforcement learning improve microscopic realism, they often overlook critical macroscopic features such as crowd density and flow, which are governed by spatio-temporal spawn dynamics, namely, when and where agents enter a scene. Traditional methods, like random spawn rates, stochastic processes, or fixed schedules, are not guaranteed to capture the underlying complexity or lack diversity and realism. To address this issue, we propose a novel approach called nTPP-GMM that models spatio-temporal spawn dynamics using Neural Temporal Point Processes (nTPPs) that are coupled with a spawn-conditional Gaussian Mixture Model (GMM) for agent spawn and goal positions. We evaluate our approach by orchestrating crowd simulations of three diverse real-world datasets with nTPP-GMM. Our experiments demonstrate the orchestration with nTPP-GMM leads to realistic simulations that reflect real-world crowd scenarios and allow crowd analysis.

## Code & Data
The code will be released soon.

## Citation
If you find our work useful, please cite our paper:

```
@inproceedings{kreutz2025whenever,
  author    = {Thomas Kreutz and Max Mühlhäuser and Alejandro Sanchez Guinea},
  title     = {Whenever, Wherever: Towards Orchestrating Crowd Simulations with Spatio-Temporal Spawn Dynamics},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```

## License
TBA
