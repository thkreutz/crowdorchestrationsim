# -*- coding: utf-8 -*-
import env.crowd_simulation_env as tk_env
import pygame
from src.utils.constants import sdd_scenes
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from collections import Counter
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import copy


import argparse

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
   

    #### Chose dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',  type=str, default='forum', help="name of the run")
    parser.add_argument('--scene', '-s', type=str, default="forum")
    parser.add_argument('--prep_dataset', '-p', type=str, default="Forum")
    parser.add_argument('--scaling', '-sc',  type=int, default=1, help="scaling factor for the map, required for, e.g., GC")
    parser.add_argument('--n_rays', '-ra',  type=int, default=32, help="number of lider rays for observation")
    parser.add_argument('--render', '-r', type=int, default=1)
    parser.add_argument('--marked', '-mr', type=int, default=1)
    parser.add_argument('--save', '-sa',  type=int, default=0, help="If run should be saved")
    parser.add_argument('--eval_run', '-e',  type=int, default=0, help="If to run a bunch of params")
    parser.add_argument('--mod_seq', '-mo',  type=int, default=0, help="If it is a modified ppt sequence..")

    args = parser.parse_args()

    dataset = args.dataset
    scene = args.scene
    prep_dataset = args.prep_dataset

    # Min max scaler for actions must be specified beforehand in this script as there is nothing saved. 
    scaler = MinMaxScaler((-1,1))
    scales = torch.load("../data_preprocessed/%s/%s_actionscales.pt" % (prep_dataset, scene))
    scaler.fit(scales)
    
    #ppt_frames = torch.load("ppt_simulation_test_long_50w.pt")

    if args.marked == 2:
        ppt_frames = torch.load("../ppt_seqs/%s/%s_marked_seq.pt" % (prep_dataset, scene))
        out_crowd = "mtpp"
    elif args.marked == 0:
        render = args.render
        ppt_frames = torch.load("../ppt_seqs/%s/%s_poisson_seq.pt" % (prep_dataset, scene))
        out_crowd = "poisson"
    else:

        if args.eval_run:
            render = False
            ppt_frames_dict = torch.load("../ppt_seqs/%s/%s_ppt_seqs_dict.pt" % (prep_dataset, scene)) # load a dict of sequences indexed by params...
            ppt_frames = []
        elif args.mod_seq:
            render = args.render
            ppt_frames = torch.load("../ppt_seqs/GC/modify/ppt_seq.pt")
        else:
            render = args.render
            ppt_frames = torch.load("../ppt_seqs/%s/%s_ppt_seq.pt" % (prep_dataset, scene))
        out_crowd = "tpp"

    print("Setting up crowd simulation, arrival process=%s.." % (out_crowd))

    model = torch.load("../models/%s/bc_%s.pth" % (prep_dataset, scene)).to("cpu")

    min_agent_id = 10 # required so that we correspond to the environment.
    
    scale_factor = args.scaling
    if render:
        rendering = "human"
    else:
        rendering = ""

    if args.dataset == "forum":
        scene = "14Jul" # can use other scenes but we used this
    else:
        scene = args.scene

    env = tk_env.CrowdSimEnv(dataset_name=dataset, scene_name=scene, data_root="", scale_factor=scale_factor, 
                             action_scaler=scaler, n_rays=args.n_rays, rendering=rendering)
    _, _ = env.reset(agents_start=[], agents_goal=[], aids=[], spawn_ids=[], destination_ids=[])
    
    if render:
        env.init_render()

    print("Press enter to start the simulation... len_simulation=%s" % len(ppt_frames))
    dec = input()
    

    if not os.path.exists("../evaluation/%s" % args.prep_dataset):
        print("Creating eval folder @ ../evaluation/%s" % args.prep_dataset)
        os.makedirs("../evaluation/%s" % args.prep_dataset)
    #### Sample sequence of spawns for each spawn point using TPP
    ## Check at each tick, if tick matches, spawn agent.
    if args.eval_run:

        for params, ppt_frames_seq in ppt_frames_dict.items():
            for j in range(len(ppt_frames_seq)):
                ppt_frames = ppt_frames_seq[j]
                # init agent id counter.

                #max_steps = 1000
                max_steps = len(ppt_frames)
                
                #max_steps = 200

                aids = min_agent_id
                
                steps = 0

                # sample agents
                agent_samples = ppt_frames[steps]
                
                # init with no agents.
                if len(agent_samples) > 0:
                    agent_samples = np.vstack(ppt_frames[steps])
                    new_aids = np.arange(aids, aids + len(agent_samples))
                    aids += len(new_aids)

                    obs, _ = env.reset(agents_start=agent_samples[:, :2], agents_goal=agent_samples[:, 2:4], aids=new_aids, spawn_ids=agent_samples[:, 4].astype(int), destination_ids=agent_samples[:, 5].astype(int))
                else:
                    obs, _ = env.reset(agents_start=[], agents_goal=[], aids=[], spawn_ids=[], destination_ids=[])

                n_agents = len(env.agents)

                print("agent_ids=%s", [ag.id for ag in env.agents])
                print("Number of agents in the crowd to be simulated...", n_agents)
                record_trajs = True
                terminated = False

                while steps < max_steps:
                    env.steps = env.steps + 1

                    if render:
                        env.clock.tick(25)

                    if len(obs) > 0:
                        # Use model to control each agent separately.
                        with torch.no_grad():
                            actions, _, _ = model(torch.Tensor(obs))

                        obs, _, terminated, _, info = env.step(actions, record_trajs, max_steps)

                    #if False:
                    agent_samples = ppt_frames[steps]
                    # update the crowd, compute observations again...
                    if len(agent_samples) > 0:
                        agent_samples = np.vstack(ppt_frames[steps])
                        new_aids = np.arange(aids, aids + len(agent_samples))
                        aids += len(new_aids)

                        obs = env.update_crowd(agents_start=agent_samples[:, :2], agents_goal=agent_samples[:, 2:4], aids=new_aids, 
                                                                                spawn_ids=agent_samples[:, 4].astype(int), destination_ids=agent_samples[:, 5].astype(int))
                    if render:
                        env.render()
                        pygame.event.pump()
                    
                    steps += 1
                    if terminated:
                        break
                
                
                # Save all the recorded trajectories for visualization and statistics.

                if args.save:
                    print("Saving crowd history...")
                    crowd_history = {ag.id : ag.trajectory for ag in env.done_agents}
                    ### (wsize, overlap, n_round, len_gen)
                    torch.save(crowd_history, "../evaluation/%s/crowd_history_%s_%s_%s_%s_%s_%s_%s.pt" % (args.prep_dataset, args.scene, out_crowd, params[0], params[1], params[2], params[3], j))

                print(info)

    else:
        # init agent id counter.
        max_steps = len(ppt_frames)
        
        #max_steps = 10000

        aids = min_agent_id
        while not dec == "q":
            steps = 0

            # sample agents
            agent_samples = ppt_frames[steps]
            
            # init with no agents.
            if len(agent_samples) > 0:
                agent_samples = np.vstack(ppt_frames[steps])
                new_aids = np.arange(aids, aids + len(agent_samples))
                aids += len(new_aids)

                obs, _ = env.reset(agents_start=agent_samples[:, :2], agents_goal=agent_samples[:, 2:4], aids=new_aids, spawn_ids=agent_samples[:, 4].astype(int), destination_ids=agent_samples[:, 5].astype(int))
            else:
                obs, _ = env.reset(agents_start=[], agents_goal=[], aids=[], spawn_ids=[], destination_ids=[])

            n_agents = len(env.agents)

            print("agent_ids=%s", [ag.id for ag in env.agents])
            print("Number of agents in the crowd to be simulated...", n_agents)
            record_trajs = True
            terminated = False

            while steps < max_steps:
                #print(steps, max_steps)
                env.steps = env.steps + 1
                if render:
                    env.clock.tick(25)

                if len(obs) > 0:
                    # Use model to control each agent separately.
                    with torch.no_grad():
                        actions, _, _ = model(torch.Tensor(obs))

                    obs, _, _, _, info = env.step(actions, record_trajs, max_steps)

                #if np.random.uniform(0,1) > 0.9:
                # Sample new agents

                #if False:
                agent_samples = ppt_frames[steps]
                # update the crowd, compute observations again...
                if len(agent_samples) > 0:
                    agent_samples = np.vstack(ppt_frames[steps])
                    new_aids = np.arange(aids, aids + len(agent_samples))
                    aids += len(new_aids)

                    obs = env.update_crowd(agents_start=agent_samples[:, :2], agents_goal=agent_samples[:, 2:4], aids=new_aids, 
                                                                            spawn_ids=agent_samples[:, 4].astype(int), destination_ids=agent_samples[:, 5].astype(int))

                if render:
                    env.render()
                    pygame.event.pump()
                
                steps += 1
            
            # Save all the recorded trajectories for visualization and statistics.
            if args.save:
                print("Saving crowd history...")
                crowd_history = {ag.id : ag.trajectory for ag in env.done_agents}

                if args.mod_seq:
                    torch.save(crowd_history, "../ppt_seqs/GC/modify/crowd_history.pt")
                else:
                    torch.save(crowd_history, "../evaluation/%s/crowd_history_%s_%s_%s.pt" % (args.prep_dataset, args.scene, out_crowd, args.marked))

            print(info)            
            print("Simulation finished. \r\n Restart (r) --- New scenario (s) --- Quit (q)")

            dec = input()
            
                
