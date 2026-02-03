import numpy as np
from imitation.data import rollout
from sklearn.preprocessing import MinMaxScaler
import argparse
import env.single_agent_env as tk_env
import importlib
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
import os

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

#trajs = pd.read_pickle("../data_preprocessed/ETH/eth_df.pkl")
#trajs = pd.read_pickle("../data_preprocessed/ETH/hotel_df.pkl")
#trajs = pd.read_pickle("../data_preprocessed/SDD/hyang_video4_df.pkl")
trajs = pd.read_pickle("../data_preprocessed/GC/gc_df.pkl")

trajs_dict = { aid : df for aid, df in list(trajs.groupby("agent_id")) }
trajs_dict.keys()


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
    parser.add_argument('--prep_dataset', '-p', type=str, default="GC")
    parser.add_argument('--epochs', '-e',  type=int, default=1000, help="number of epochs")
    parser.add_argument('--batch_size', '-b',  type=int, default=1024, help="batch size")
    parser.add_argument('--scaling', '-sc',  type=int, default=1, help="scaling factor for the map, required for, e.g., GC")
    parser.add_argument('--rays', '-r',  type=int, default=32, help="number of lider rays for observation")
 

    args = parser.parse_args()

    scale_factor = args.scaling
    rays = args.rays

    print("Preparing env.. ")
    scale_factors_path = "../data_preprocessed/%s/%s_actionscales.pt" % (args.prep_dataset, args.scene)
    
    scales = torch.load(scale_factors_path)
    scaler = MinMaxScaler((-1,1))
    scaler.fit(scales)

    if args.dataset == "forum":
        scene = "14Jul" # just a dummy scene so that we get env action and observation space.
    else:
        scene = args.scene

    env = tk_env.CustomEnv(dataset_name=args.dataset, scene_name=scene, data_root="", 
                           scale_factor=scale_factor, n_agents=1, rendering="", n_rays=rays, action_scaler=scaler)

    print("Loading rollouts..")
    # Load rollouts
    rollout_path = "../data_preprocessed/%s/%s_rollouts_r%s_s%s.pt" % (args.prep_dataset, args.scene, args.rays, args.scaling)
    rollouts = torch.load(rollout_path)
    transitions = rollout.flatten_trajectories(rollouts)
    #transitions = torch.Tensor(transitions).to("cuda")

    print("init model..")
    rng = np.random.default_rng()
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        batch_size=args.batch_size,
        rng=rng,
        device="cpu"
    )

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward before training: {reward_before_training}")

    print("Start training...")
    bc_trainer.train(n_epochs=args.epochs)

    if not os.path.exists("../models/%s/" % args.prep_dataset):
        os.makedirs("../models/%s/" % args.prep_dataset)
    output_path = "../models/%s/bc_%s.pth" % (args.prep_dataset, args.scene) 
    print("Finished training... saving model to %s" % output_path)

    torch.save(bc_trainer.policy, output_path)

    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")


