# Whenever, Wherever: Towards Orchestrating Crowd Simulations with Spatio-Temporal Spawn Dynamics

Public Repository for our ICRA 2025 accepted paper: Whenever, Wherever: Towards Orchestrating Crowd Simulations with Spatio-Temporal Spawn Dynamics. 

[Project Page](https://thkreutz.github.io/projects/crowdorchestra.html) - [Paper](https://ieeexplore.ieee.org/document/11128302)

## Authors
Thomas Kreutz, Max Mühlhäuser, and Alejandro Sanchez Guinea

## Abstract
Realistic crowd simulations are essential for immersive virtual environments, relying on both individual behaviors (microscopic dynamics) and overall crowd patterns (macroscopic characteristics). While recent data-driven methods like deep reinforcement learning improve microscopic realism, they often overlook critical macroscopic features such as crowd density and flow, which are governed by spatio-temporal spawn dynamics, namely, when and where agents enter a scene. Traditional methods, like random spawn rates, stochastic processes, or fixed schedules, are not guaranteed to capture the underlying complexity or lack diversity and realism. To address this issue, we propose a novel approach called nTPP-GMM that models spatio-temporal spawn dynamics using Neural Temporal Point Processes (nTPPs) that are coupled with a spawn-conditional Gaussian Mixture Model (GMM) for agent spawn and goal positions. We evaluate our approach by orchestrating crowd simulations of three diverse real-world datasets with nTPP-GMM. Our experiments demonstrate the orchestration with nTPP-GMM leads to realistic simulations that reflect real-world crowd scenarios and allow crowd analysis.

## Code & Data
Environment:
- conda env create -f environment.yml
- conda activate imcrowds

Step 1: Download data from OpenTraj https://github.com/crowdbotp/OpenTraj
- ETC, GC, Edinburgh
- (Supports overall ETH/UCY, SDD, GC, Edinburgh Forum, ATC)

Step 2: Data preprocessing
- data.ipynb holds all the code to preprocess each dataset, respectively

Step 3: fit nTPP and generate ntpp+poisson sequences (replace with respective dataset shortcut, example for Forum below)
- python fit_sample_ntppgmm.py -d Forum -s 14Jul -de cpu

Step 4: agent policy training (BC) (replace with respective dataset shortcut, example for Forum below)
- python prep_imitation_rollouts.py -d forum -s "14Jul" -sc 1 -o Forum -r 32
- python train_bc.py -d forum -s 14Jul -p Forum -sc 1 -r 32 -e 100

Step 5: Orchestrated Crowd Simulation (replace with respective dataset shortcut, example for Forum below)
- python tpp_crowd_sim.py -d forum -s 14Jul -p Forum -sc 1 -ra 32 -r 1 -mr 1 -sa 1 -e 0 -mo 0

Other 1: Test policy that controls all agents with spawns and timings from real dataset:
- python crowd_sim.py 

Other 2: Test policy controlling only a single agent in a sequence from the real dataset:
- python single_agent_sim.py

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
