# -*- coding: utf-8 -*-
import env.crowd_simulation_env as tk_env
import pygame
from src.utils.constants import sdd_scenes
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import copy


class CrowdEmission:
    
    def __init__(self, trajs, coord_transform, n_samples=100, scaling=1):
        
        self.n_samples = n_samples
        self.scaling = scaling
        self.coord_transform = coord_transform
        
        starts = []
        goals = []
        self.frames = []
        trajectories = []
        for i, df in trajs.items():
            starts.append(df[["start_x", "start_y"]].values[0])
            goals.append(df[["goal_x", "goal_y"]].values[0])
            self.frames.append(df.frame_id.values[0])
            trajectories.append(df[["pos_x", "pos_y"]].values)
            
        self.starts_dataset = np.vstack(starts)
        self.goals_dataset = np.vstack(goals)
        
        # Run clustering on the positions to find start-goal pairs and compute respective pair statistics for sampling
        self.start_goal_clustering()
        self.cooccurence()
    
    def scene_probas(self, world_positions):
        #positions = np.array(list((agents_positions.values())))[:, 1:3]
        positions = copy.deepcopy(world_positions)

        # positions_pixel = coord_transform.get_pixel_positions(positions)

        dbsc = DBSCAN(eps=.2, min_samples=10).fit(positions)
        #dbsc = DBSCAN(eps=1, min_samples=2).fit(positions)
        positions = positions[dbsc.labels_ != -1]
        
        goal_clusters = dbsc.labels_[dbsc.labels_ != -1]
        goal_areas = {}
        for k in range(len(set(goal_clusters))):
            temp = positions[goal_clusters == k]
            #print(temp)
            goal_areas[k] = temp

        stats = [(torch.Tensor(np.mean(goal_areas[k], axis=0)), torch.Tensor(np.cov(goal_areas[k].T))) for k in goal_areas.keys()]
        return goal_areas, stats, dbsc.labels_

    def sample_agents(self, stats, n_agents):
        estimates = [torch.Tensor(np.random.multivariate_normal(p[0], p[1], n_agents)) for p in stats]
        return estimates
    
    def start_goal_clustering(self):
        self.agent_starts, self.start_stats, self.cluster_preds_start = self.scene_probas(self.starts_dataset)
        self.agent_deaths, self.death_stats, self.cluster_preds_death = self.scene_probas(self.goals_dataset)
        # Sample agents
        
        self.starts = self.sample_agents(self.start_stats, self.n_samples)
        self.deaths = self.sample_agents(self.death_stats, self.n_samples)
    
    def cooccurence(self):
        #### Find cluster co-occurences

        cluster_combinations = np.column_stack([self.cluster_preds_start, self.cluster_preds_death])
        # get top most common pairs
        # collect top X most common where cluster not -1 and they are different
        most_common_pairs = Counter(list(map(tuple, cluster_combinations))).most_common()
        self.start_goal_pairs = []
        self.start_goal_pair_frequency = []
        for a, freq in most_common_pairs:
            if a[0] != a[1] and (a[0] != -1) and (a[1] != -1):
                self.start_goal_pairs.append(a)
                self.start_goal_pair_frequency.append(freq)
            if len(self.start_goal_pairs) == 30:
                break
    
    def sample(self):
        gen_agents = []
        #for a in range(n_agents):
        #    pair = np.random.choice(np.arange(len(start_goal_pairs)))
        #    start_cluster, death_cluster = start_goal_pairs[pair]
        for start_cluster, death_cluster in self.start_goal_pairs:   
            for _ in range(5):
                start_pos = self.starts[start_cluster][np.random.randint(0, self.n_samples)]
                death_pos = self.deaths[death_cluster][np.random.randint(0, self.n_samples)]
                gen_agents.append(torch.cat([start_pos, death_pos]))
    
        gen_agents = np.vstack(gen_agents)
        return gen_agents
    
    def temporal_emission_sampling(self, step):
        gen_agents = []
        
        for start_cluster, death_cluster in self.start_goal_pairs:   
            
            # Can use a simply statistical rule here
            # => i.e. random thresh sampling from each cluster pair
            
            # Or try a more sophisticated model.
            # => weighted sampling for each pair
            # => Sequence model like HMM with emission probabilities
            # => stochastic process
            # => include other statistics for sampling a number of agents
            # => Catch collisions between agents and agents environment at start point.
            
            #if np.random.uniform(0,1) > 0.8:
            for i in range(np.random.randint(3)):
                start_pos = self.starts[start_cluster][np.random.randint(0, self.n_samples)]
                death_pos = self.deaths[death_cluster][np.random.randint(0, self.n_samples)]
                gen_agents.append(torch.cat([start_pos, death_pos]))

        return np.vstack(gen_agents)



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

    #trajs = pd.read_pickle("../data_preprocessed/ETH/eth_df.pkl")
    #trajs = pd.read_pickle("../data_preprocessed/ETH/hotel_df.pkl")
    #trajs = pd.read_pickle("../data_preprocessed/SDD/hyang_video4_df.pkl")
   
    # Min max scaler for actions must be specified beforehand in this script as there is nothing saved. 
    scaler = MinMaxScaler((-1,1))

    ### eth
    #ma = np.array([15, 35])
    #mi = np.array([-16, -30])
    
    ### gc bounds for minmax scaler.
    #ma = np.array([37.5,  18]) 
    #mi = np.array([ -28.5, -126.5 ])

    ma = np.array([259.25, 133.5 ])
    mi = np.array([ -308.5, -239.5 ])


    scaler.fit(np.vstack([ma, mi]))
    
    scale_factor = 2
    env = tk_env.CrowdSimEnv(dataset_name="gc", scene_name="gc", data_root="", scale_factor=scale_factor, action_scaler=scaler, rendering="human")
    _, _ = env.reset(agents_start=[], agents_goal=[], aids=[])
    env.init_render()

    print("Setting up crowd emission controller..")
    crowd_ems = CrowdEmission(env.trajs, env.coord_transform, 250)
    min_agent_id = 10 # required so that we correspond to the environment.

    #model = torch.load("bc_gc.pth")
    model = torch.load("bc_gc_full_500epochs.pth").to("cpu")

    
    print("Press enter to start the simulation...")
    dec = input()
    

    #### Sample sequence of spawns for each spawn point using TPP
    ## Check at each tick, if tick matches, spawn agent.

    # init agent id counter.
    aids = min_agent_id
    while not dec == "q":
        
        # sample agents
        agent_samples = crowd_ems.temporal_emission_sampling(1)
        new_aids = np.arange(aids, aids + len(agent_samples))
        aids += len(agent_samples)

        # init with no agents.
        if len(agent_samples) > 0:
            obs, _ = env.reset(agents_start=agent_samples[:, :2], agents_goal=agent_samples[:, 2:], aids=new_aids)
        else:
            obs, _ = env.reset(agents_start=[], agents_goal=[], aids=[])

        n_agents = len(env.agents)

        print("agent_ids=%s", [ag.id for ag in env.agents])
        print("Number of agents in the crowd to be simulated...", n_agents)
        steps = 0
        while True:
            
            env.clock.tick(25)

            if len(obs) > 0:
                # Use model to control each agent separately.
                with torch.no_grad():
                    actions, _, _ = model(torch.Tensor(obs))

                obs, _, terminated, _, info = env.step(actions)

            env.render()
            pygame.event.pump()

            #pygame.image.save(env.window, "../demo_visualization/imcrowdgail_%s.png" % steps)

            if steps % 50 == 0:

            #if np.random.uniform(0,1) > 0.9:
                # Sample new agents
                agent_samples = crowd_ems.temporal_emission_sampling(1)
                new_aids = np.arange(aids, aids + len(agent_samples))
                aids += len(agent_samples)
                
                # update the crowd, compute observations again...
                if len(agent_samples) > 0:
                    obs = env.update_crowd(agents_start=agent_samples[:, :2], agents_goal=agent_samples[:, 2:], aids=new_aids)
                print("agent_ids=", [ag.id for ag in env.agents])
            steps += 1
            if terminated:
                break

        print(info)            
        print("Simulation finished. \r\n Restart (r) --- New scenario (s) --- Quit (q)")
        
        dec = input()
        
            
