from tqdm import tqdm
import env.single_agent_env as tk_env
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import copy
import pandas as pd
import numpy as np
from src.tpp.tpp import NeuralTPP

class CrowdEmission:
    
    def __init__(self, trajs, coord_transform, n_samples=250, scaling=1, dbscan_eps=1, dbscan_min_samples=5, top_k=50):
        
        self.n_samples = n_samples
        self.scaling = scaling
        self.coord_transform = coord_transform
        self.top_k = top_k

        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
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
        
        #### Spawn Emission Sequences

        #print(self.start_goal_pairs)
        #print(np.unique(self.cluster_preds_start) )
        frame_df = pd.DataFrame(np.column_stack([self.frames, self.cluster_preds_start, np.ones(len(self.frames)).astype(int)]), 
                                    columns=["frame_id", "cluster", "counts"])

        #frame_df = pd.DataFrame(np.column_stack([self.frames, self.cluster_preds_start, self.cluster_preds_death, np.ones(len(self.frames)).astype(int)]), 
         #                           columns=["frame_id", "cluster", "goal", "counts"])
                                              
        cluster_spawns = {i : df for i, df in list(frame_df.groupby("cluster")) if i in np.array(self.start_goal_pairs)[:,0]}
        # get frame emission sequence df for each cluster
        self.emission_sequences = {}

        for i, df in cluster_spawns.items():
            # count the frames
            df_group = df.groupby("frame_id")["counts"].count().reset_index()
            # make new df which fills the frames to a full sequence
            # we take the max frames of the dataset

            #df_temp = pd.DataFrame(np.arange(max(df.frame_id.values)+1), columns=["frame_id"])
            df_temp = pd.DataFrame(np.arange(max(self.frames)+1), columns=["frame_id"])
            # merge and fill with 0s where no emission happened
            result_df = pd.merge(df_temp, df_group, on='frame_id', how='left')
            result_df['counts'] = result_df['counts'].fillna(0).astype(int)

            self.emission_sequences[i] = result_df
    
    def scene_probas(self, world_positions):
        #positions = np.array(list((agents_positions.values())))[:, 1:3]
        positions = copy.deepcopy(world_positions)

        # positions_pixel = coord_transform.get_pixel_positions(positions)
        
        #### GC
        #dbsc = DBSCAN(eps=.2, min_samples=10).fit(positions)
        
        #### ATC
        dbsc = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(positions)

        
        #dbsc = DBSCAN(eps=1, min_samples=2).fit(positions)
        positions = positions[dbsc.labels_ != -1]
        
        goal_clusters = dbsc.labels_[dbsc.labels_ != -1]
        goal_areas = {}
        for k in range(len(set(goal_clusters))):
            temp = positions[goal_clusters == k]
            #print(temp)
            goal_areas[k] = temp

        stats = [(np.mean(goal_areas[k], axis=0), np.cov(goal_areas[k].T)) for k in goal_areas.keys()]
        return goal_areas, stats, dbsc.labels_

    def sample_agents(self, stats, n_agents):
        estimates = [np.random.multivariate_normal(p[0], p[1], n_agents) for p in stats]
        return estimates
    
    def start_goal_clustering(self):
        self.agent_starts, self.start_stats, self.cluster_preds_start = self.scene_probas(self.starts_dataset)
        self.agent_deaths, self.death_stats, self.cluster_preds_death = self.scene_probas(self.goals_dataset)
        # Sample agents
        
        self.starts = self.sample_agents(self.start_stats, self.n_samples)
        self.deaths = self.sample_agents(self.death_stats, self.n_samples)
    
    ### For very interactive sim, we update the stats for spawn/exit clusters here. 
    ### Further, specify spread
    ### Further, add NEW spawns
    ### event further, MANUALLY specify them
    ###def update_stats(self)
 
    
    def cooccurence(self):
        #### Find cluster co-occurences

        cluster_combinations = np.column_stack([self.cluster_preds_start, self.cluster_preds_death])
        # get top most common pairs
        # collect top X most common where cluster not -1 and they are different
        most_common_pairs = Counter(list(map(tuple, cluster_combinations))).most_common()
        self.start_goal_pairs = []
        self.start_goal_pair_frequency = []
        for a, freq in most_common_pairs:
            
            # if a[0] != a[1] and (a[0] != -1) and (a[1] != -1):
            if (a[0] != -1) and (a[1] != -1):
                self.start_goal_pairs.append(a)
                self.start_goal_pair_frequency.append(freq)
            if len(self.start_goal_pairs) == self.top_k:
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
                gen_agents.append(np.column_stack([start_pos, death_pos]))
    
        gen_agents = np.vstack(gen_agents)
        return gen_agents
    
    def sample_from_pair(self, pair):
        samples = []
        for start_cluster, death_cluster in pair:
            start_pos = self.starts[start_cluster][np.random.randint(0, self.n_samples)]
            death_pos = self.deaths[death_cluster][np.random.randint(0, self.n_samples)]
            samples.append(np.hstack([start_pos, death_pos]))
            
        return np.vstack(samples)
    
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
                gen_agents.append(np.hstack([start_pos, death_pos]))

        return np.vstack(gen_agents)


class SpawnTPPs:
    
    def __init__(self, spawn_emissions, window_size, overlap):
        # Specify window size and overlap
        self.spawn_emissions = spawn_emissions
        self.spawn_tpps = {}
        self.spawn_windows = {}
        
        self.window_size = window_size
        self.overlap = overlap
        
    def preprocess_windows(self, spawn_cluster):
       # window_size = 100
       # overlap = 5

        # Initialize an empty list to store the sliding windows
        windows = []

        # Iterate over the DataFrame to create sliding windows
        for i in range(0, len(self.spawn_emissions[spawn_cluster]), self.overlap):
            window = self.spawn_emissions[spawn_cluster][ ['frame_id', 'counts']].iloc[i:i+self.window_size].values

            # only keep frame where count > 0
            window = window[window[:,1] > 0]
            
            if len(window) > 0:
                # normalize arrival time times to max window size
                window[:,0] = window[:,0] - window[0][0]

                # only keep arrivals, not the counts
                windows.append(window[:,0])
            

        self.spawn_windows[spawn_cluster] = windows
        
        return windows
    
    def preprocess_training(self, windows, device="cuda"):
        arrival_times_list = windows  ### window length must be scaled to 50
        # t_end = length of the observerd time interval [0, t_end]
        t_end = self.window_size
        seq_lengths = torch.tensor([len(t) for t in arrival_times_list], dtype=torch.long).to(device)

        def get_inter_times(t, t_end):
            tau = np.diff(t, prepend=0.0, append=t_end)
            return torch.tensor(tau, dtype=torch.float32, device=device)

        inter_times_list = [get_inter_times(t, t_end) for t in arrival_times_list]
        inter_times = pad_sequence(inter_times_list, batch_first=True)
        
        return t_end, seq_lengths, inter_times
    
    def train_ppt(self, spawn_cluster_id, inter_times, seq_lengths, t_end, max_epochs=500, verbose=False, device="cuda"):
        model = NeuralTPP().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        
        loss_hist = []
        for epoch in range(max_epochs + 1):
            opt.zero_grad()
            loss = model.nll_loss(inter_times, seq_lengths).mean() / t_end
            loss.backward()
            opt.step()
            
            if verbose:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: loss = {loss.item():.3f}")
            loss_hist.append(loss.item())
        
        self.spawn_tpps[spawn_cluster_id] = model
        
        return model, loss_hist
    
    def sample_sequences(self, spawn_cluster_id, n_samples, t_end, device="cuda"):
        with torch.no_grad():
            gen_inter_times, gen_seq_lengths = self.spawn_tpps[spawn_cluster_id].sample(n_samples, t_end, device=device)

        gen_arrival_times = gen_inter_times.cumsum(-1)
        generated_sequences = []
        for i in range(gen_arrival_times.shape[0]):
            
            t = gen_arrival_times[i, :gen_seq_lengths[i]].cpu().numpy()
            generated_sequences.append(t)
        return generated_sequences


def fit_crowd_emissions(dataset_name="gc", scene_name="gc", 
                            scale_factor=1, dbscan_eps=1, dbscan_min_samples=5, top_k=50):

    # Need the env
    env = tk_env.CustomEnv(dataset_name=dataset_name, scene_name=scene_name, data_root="", scale_factor=1, 
                        n_agents=1, rendering="", action_scaler=None)

        #trajs, coord_transform, n_samples=250, scaling=1, dbscan_eps=1, dbscan_min_samples=5, top_k=50
    crowd_ems = CrowdEmission(env.trajs, env.coord_transform, 
                            n_samples=250, scaling=scale_factor, dbscan_eps=dbscan_eps, 
                            dbscan_min_samples=dbscan_min_samples, top_k=top_k)

    # Return, can use this object later to evaluate the clusters.
    return crowd_ems, env

def fit_tpps_on_dataset(crowd_ems, wsize=100, overlap=5, max_epochs=500, device="cuda"):

    #wsize = 100
    spawn_tpps = SpawnTPPs(crowd_ems.emission_sequences, window_size=wsize, overlap=overlap)
    for k in tqdm(spawn_tpps.spawn_emissions.keys()):
        #print("Generating tpp for spawn=%s" % k)
        windows = spawn_tpps.preprocess_windows(k)
        t_end, seq_lengths, inter_times = spawn_tpps.preprocess_training(windows,device=device)
        _, _ = spawn_tpps.train_ppt(k, inter_times, seq_lengths, t_end, max_epochs=max_epochs, verbose=False, device=device)

    ### Co-Occurence frequency distribution
    spawns = {}
    i = 0
    for start, destination in crowd_ems.start_goal_pairs:
        
        if not start in spawns:
            spawns[start] = []
            spawns[start].append( [destination, crowd_ems.start_goal_pair_frequency[i]] )
        else:
            spawns[start].append( [destination, crowd_ems.start_goal_pair_frequency[i]] )
            
        i += 1
        
    for k, v in spawns.items():
        spawns[k] = np.vstack(v)
        
        ## add new column
        freqs = spawns[k][:,1]
        probs = freqs / np.sum(freqs)
        
        spawns[k] = np.column_stack( [spawns[k], probs] )

    # return
    return spawn_tpps, spawns


def generate_sequences(spawn_tpps, spawns, crowd_ems, n_rounds=10, len_gen=1000, device="cuda"):
        
    spawn_simulation = {}
    #n_rounds = 10
    #len_gen = 1000
    for k in tqdm(spawn_tpps.spawn_emissions.keys()):
        windows_test = spawn_tpps.sample_sequences(k, n_rounds, len_gen, device=device)
        spawn_simulation[k] = np.concatenate([windows_test[i] + (i*len_gen) for i in range(len(windows_test))]).astype(int)
    
    
    # Now we have a sequence of frames for each spawn.
    # What we have to do now, is count the number of occurences at each frame (because we are discrete, not continuous.)
    # Tiny hack to get multiple agents at the same time.

    # get frame emission sequence df for each cluster
    n_max_frames = n_rounds * len_gen
    simulated_emission_sequences = {}
    for k, frames in spawn_simulation.items():
        df = pd.DataFrame(np.column_stack( [frames, np.ones(len(frames)).astype(int)]), columns=["frame_id", "counts"])
        # count the frames
        df_group = df.groupby("frame_id")["counts"].count().reset_index()
        # make new df which fills the frames to a full sequence
        # we take the max frames of the dataset
        
        #df_temp = pd.DataFrame(np.arange(max(df.frame_id.values)+1), columns=["frame_id"])
        df_temp = pd.DataFrame(np.arange(n_max_frames+1), columns=["frame_id"])
        # merge and fill with 0s where no emission happened
        result_df = pd.merge(df_temp, df_group, on='frame_id', how='left')
        result_df['counts'] = result_df['counts'].fillna(0).astype(int)
        
        simulated_emission_sequences[k] = result_df
                
    # Now, we "simulate", i.e., we spawn agents.
    # --- This could be done on the fly while simulating, but we just sample some here.
    n_frames = n_rounds * len_gen
    frame_wise_spawns = []
    for frame in range(n_frames):
        
        frame_spawns = []
        for k, df in simulated_emission_sequences.items():
            #if not k in [1, 2, 6, 9]:
            #    break
            if df.counts.values[frame] > 0:
                X = spawns[k]

                candidates = X[:,0]
                n_draws = df.counts.values[frame]
                prob_distribution = X[:,2]
                
                draws = np.random.choice(candidates, n_draws, p=prob_distribution)
                pairs = np.column_stack([np.ones(len(draws)) * k, draws]).astype(int)

                #draws = np.column_stack( [crowd_ems.sample_from_pair(pairs), np.ones(len(draws)) * k])
                draws = np.column_stack( [crowd_ems.sample_from_pair(pairs), pairs])
                
                frame_spawns.append(draws)
                
        frame_wise_spawns.append(frame_spawns)

    return simulated_emission_sequences, frame_wise_spawns