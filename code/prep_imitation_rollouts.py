import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from sklearn.preprocessing import MinMaxScaler
import argparse
import env.single_agent_env as tk_env
import importlib
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from imitation.data.types import Trajectory

def get_data(trajs_dict, idx):
    s = trajs_dict[idx][["pos_x", "pos_y"]].values
    s_pixel = env.coord_transform.get_pixel_positions(s) #/ 4.0
    actions, states = get_state_action_pairs(s_pixel)
    _, obs = env.replay_agent(s[0], s[-1], actions)#np.flip(actions, axis=1)
    return actions, s

def get_state_action_pairs(positions):
    ##### Single integrator
    # Simulation Parameters
    dt = 1  # time step
    
    # Initialize Arrays to Store Results
    linear_velocities_x = []
    linear_velocities_y = []
    
    # Init State
    state = np.array([positions[0, 0], positions[0, 1]])  # x, y
    
    # Single Integrator Dynamics Model
    def update_state(current_state, linear_velocity_x, linear_velocity_y, dt):
        x, y = current_state

        x_next = x + linear_velocity_x * dt
        y_next = y + linear_velocity_y * dt

        return np.array([x_next, y_next])
    
    states = [state]
    actions = []
    # Simulation Loop
    for i in range(len(positions) - 1):
        current_position = state
        next_position = positions[i + 1]

        # Calculate control as the difference in positions scaled by a factor
        linear_velocity_x = (next_position[0] - current_position[0]) / dt
        linear_velocity_y = (next_position[1] - current_position[1]) / dt

        # Store the actions
        actions.append([linear_velocity_x, linear_velocity_y])
        
        # Update the state using the single integrator dynamics model
        state = update_state(state, linear_velocity_x, linear_velocity_y, dt)
        states.append(state)
        
    return np.array(actions), np.array(states)

class ExpertActionScaler:
    def __init__(self, expert_actions):
        self.actions = np.vstack(expert_actions)

        ### Scale actions
        self.scaler = MinMaxScaler( (-1,1) )
        self.scaler.fit(self.actions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',  type=str, default='gc', help="name of the run")
    parser.add_argument('--scene', '-s', type=str, default="gc")
    parser.add_argument('--scaling', '-sc',  type=int, default=1, help="scaling factor for the map, required for, e.g., GC")
    parser.add_argument('--output', '-o',  type=str, default=1, help="output dataset name")
    parser.add_argument('--rays', '-r',  type=int, default=32, help="number of lider rays for observation")
    args = parser.parse_args()

    scale_factor = args.scaling
    env = tk_env.CustomEnv(dataset_name=args.dataset, scene_name=args.scene, data_root="", 
                           scale_factor=scale_factor, n_agents=1, rendering="", n_rays=args.rays, action_scaler=None)
    #env = tk_env.CustomEnv(dataset_name="gc", scene_name="gc", data_root="", scale_factor=1, n_agents=1, rendering="human"

    #### Have to run this only once.

    ### Collect all state action pairs
    
    trajs_dict = env.trajs
    print("Collecting observations from existing trajectories to get a set of rollouts...")
    actions_list = []
    obs_list = []
    rollouts = []
    for aid, traj_df in tqdm(trajs_dict.items()):
        s = traj_df[["pos_x", "pos_y"]].values
        s_pixel = env.coord_transform.get_pixel_positions(s) / scale_factor
        actions, states = get_state_action_pairs(s_pixel)
        _, obs = env.replay_agent(actions, aid) #np.flip(actions, axis=1))
        actions_list.append(actions)
        obs_list.append(obs) # we do not want the last observation

    expert_action_scaler = ExpertActionScaler(actions_list)

    #### Rollouts in imitation lib format
    rollouts = [Trajectory(obs=obs, acts=expert_action_scaler.scaler.transform(acts), infos=None, terminal=True) for obs, acts in zip(obs_list, actions_list)]
    torch.save(rollouts, "../data_preprocessed/%s/%s_rollouts_r%s_s%s.pt" % (args.output, args.scene, args.rays, args.scaling))
    
    torch.save(np.vstack([expert_action_scaler.scaler.data_max_, expert_action_scaler.scaler.data_min_]), "../data_preprocessed/%s/%s_actionscales.pt" % (args.output, args.scene))
