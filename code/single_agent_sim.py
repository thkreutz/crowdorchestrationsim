# -*- coding: utf-8 -*-
import env.single_agent_env as tk_env
import pygame
from src.utils.constants import sdd_scenes
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    
    # Init State
    state = np.array([positions[0, 0], positions[0, 1]])  # x, y
    
    # Single Integrator Dynamics Model
    def update_state(current_state, linear_velocity_x, linear_velocity_y, dt):
        x, y = current_state

        x_next = x + linear_velocity_x * dt
        y_next = y + linear_velocity_y * dt

        return np.array([x_next, y_next])
    
    # initialize arrays to store results
    states = [state]
    actions = []

    # Simulation Loop, we use this format so that other motion models could be switched out against single integrator dynamics.
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


def run_policy(env, model, aid, max_steps=300):
    # Reset env.
    #env = tk_env.CustomEnv(dataset_name="eth", scene_name="eth", data_root="", scale_factor=1, n_agents=1, rendering="human", action_scaler=expert_action_scaler.scaler)
    #env.init_render()
    print("Resetting...")
    obs,_ = env.reset(replay_agent=True, aid=aid)

    print("Start policy...")
    cum_reward = 0
    try:
        # max steps
        for i in range(max_steps):
            
            with torch.no_grad():
                action, _, _ = model(torch.Tensor(obs).unsqueeze(0)) #.to("cpu"))
                action = action[0] #.cpu()

            obs, reward, terminated, _, _ = env.step(action)
            cum_reward += reward
            if terminated:
                break

            env.render()

            # pump event queue?
            pygame.event.pump()

            #pygame.image.save(env.window, "../demo_visualization/single_bc_gc_%s_%s.png" % (aid, i) )

            env.clock.tick(10)
        print("Terminated.")
        #pygame.quit()
    except Exception as e:
        print(e)
        #pygame.quit()

    print("Cummulative reward: ", cum_reward)


if __name__ == "__main__":
    ### This script is mainly used for testing and for visualization.

    ### Functionality of the gym environment:
    # Single-agent pedestrian policy environment.
    # Multi-agent crowd replay.
    # Multi-agent simulation where each pedestrian follows its own policy.
    # Pseudo LiDAR-based observation.
    # Integration with imitation learning and stable baselines.
    # Supports a broad range of common pedestrian trajectory prediction datasets, new datasets can easily be added given the respective format.
    # Control in PIXEL space, not in world space (design choice). Actions have to be scaled accordingly to pixel space.

    #environment = tk_env.CustomEnv(n_agents=1, rendering="human")
    #environment = tk_env.CustomEnv(dataset_name="gc", scene_name="gc", data_root="", scale_factor=2, n_agents=1, rendering="human")
    #environment = tk_env.CustomEnv(dataset_name="sdd", scene_name=sdd_scenes[2], data_root="", scale_factor=4, n_agents=1, rendering="human")


    #trajs = pd.read_pickle("../data_preprocessed/ETH/eth_df.pkl")
    #trajs = pd.read_pickle("../data_preprocessed/ETH/hotel_df.pkl")
    #trajs = pd.read_pickle("../data_preprocessed/SDD/hyang_video4_df.pkl")
    #trajs = pd.read_pickle("../data_preprocessed/GC/gc_df.pkl")

    #trajs = pd.read_pickle("../data_preprocessed/Forum/Forum_df.pkl")
    #trajs = pd.read_pickle("../data_preprocessed/ATC/atc_df.pkl")
  

    min_agent_id = 10

    # Min max scaler for actions must be specified beforehand in this script as there is nothing saved. 
    scaler = MinMaxScaler((-1,1))

    # eth
    #ma = np.array([15, 35])
    #mi = np.array([-16, -30])

    # gc
    #ma = np.array([37.5,  18]) 
    #mi = np.array([ -28.5, -126.5 ])

    #ma = np.array([259.25, 133.5 ])
    #mi = np.array([ -308.5, -239.5 ])
    #scaler.fit(np.vstack([ma, mi]))

    scales = torch.load("../data_preprocessed/ETH/eth_actionscales.pt")
    scaler.fit(scales)

    #scales = torch.load("../data_preprocessed/Forum/forum_actionscales.pt")
    #scaler.fit(scales)

    #scales = torch.load("../data_preprocessed/Forum/forum_actionscales.pt")
    #scaler.fit(scales)

    #scales = torch.load("../data_preprocessed/ATC/atc_actionscales.pt")
    #scaler.fit(scales)

    
    scale_factor = 1
    env = tk_env.CustomEnv(dataset_name="eth", scene_name="eth", data_root="", scale_factor=1, n_agents=1, action_scaler=scaler, rendering="human")
    #env = tk_env.CustomEnv(dataset_name="sdd", scene_name=("hyang", 4), data_root="", scale_factor=1, n_agents=1, action_scaler=scaler, rendering="human")
    
    #env = tk_env.CustomEnv(dataset_name="forum", scene_name="22Jul", data_root="", scale_factor=scale_factor, n_rays=32, n_agents=1, action_scaler=scaler, rendering="human")
    #env = tk_env.CustomEnv(dataset_name="atc", scene_name="atc", data_root="", scale_factor=scale_factor, n_rays=32, n_agents=1, action_scaler=scaler, rendering="human")
    
    env.init_render()

    simulation_mode = "replay"
    #model = torch.load("gail_gc_300k.pth").to("cpu")
    #model = torch.load("bc_gc_full.pth").to("cpu")

    #model = torch.load("bc_gc_full.pth").to("cpu")

    #model = torch.load("bc_gc_full_500epochs.pth").to("cpu")

    #model = torch.load("bc_gail_5.pth").to("cpu")

    #model = torch.load("../models/ATC/bc_atc_100epochs.pth").to("cpu")
    #model = torch.load("../models/ATC/bc_atc.pth").to("cpu")

    #model = torch.load("../models/Forum/bc_forum.pth").to("cpu")

    #model = torch.load("../models/gc_gail_9.pth").to("cpu")


    #model = torch.load("../models/Forum/bc_forum.pth").to("cpu")
    #model = torch.load("../models/GC/bc_gc.pth").to("cpu")

    # eth
    model = torch.load("../models/ETH/bc_eth.pth").to("cpu")

    dec = "s"
    while not dec == "q":

        if dec == "s":
            print("Chose one of the following agent ids to start the replay=", sorted(list(env.trajs.keys())[:200] ))
            aid = int(input()) #+ min_agent_id

        if simulation_mode == "replay":
            print("replaying agent=%s" % aid)

            # This is replay.
            s = env.trajs[aid][["pos_x", "pos_y"]].values
            s_pixel = env.coord_transform.get_pixel_positions(s) // scale_factor
            actions, states = get_state_action_pairs(s_pixel)
            _, obs = env.replay_agent(actions, aid=aid, render_replay=True)
        else:
            print("controlling agent=%s" % aid)
            run_policy(env, model, aid)


        print("Simulation finished. \r\n Restart (r) --- New agent (s) --- Change simulation mode (c) --- Quit (q)")
        dec = input()

        if dec == "c":
            print("Enter <replay> or <policy>")
            simulation_mode = input()
            print("Simulation_mode updated to <%s>" % simulation_mode)

    # This is control.
    # Control only with a trained policy.
    if False:
        run = True
        while run:
            # set game speed to 30 fps
            env.clock.tick(25)   ### ETH is 2.5fps
            # ─── CONTROLS ───────────────────────────────────────────────────────────────────
            # end while-loop when window is closed
            get_event = pygame.event.get()
            for event in get_event:
                #print(event)
                if event.type == pygame.QUIT:
                    run = False
            # get pressed keys, generate action
            get_pressed = pygame.key.get_pressed()
            action = tk_env.pressed_to_action(get_pressed)
            #print(action)
            # calculate one step

            #next_actions = []
            #for a in environment.agents:
            #    next_actions.append(a.policy_step())

            #next_actions = [action] * len(environment.agents)

            history_observations, reward, done, b, info = env.step(action)
            #print(len(history_observations[-1]))
            print("internals: ", history_observations[:6])
            print("externals: ", history_observations[6:20])
            print("reward:", reward)
            
            if done:
                print("done")
                break
            #print(history_observations.shape)
            # render current state
            env.render()
            # pump event queue?
            pygame.event.pump()

        pygame.quit()
