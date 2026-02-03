from torch_geometric.data import Dataset
import torch
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.data.background import BackgroundObservations
import os
import glob
from collections import defaultdict

class EgoTransform(object):

    """ Transform the 

    """ 

# based on https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# skip download, transform, and pre_transform
class TrajectoryGraphDataset(Dataset):
    def __init__(self, dfs, background_image_path, homography_mat_path, lidar_num_rays=64, lidar_max_dist=300, lidar_fov=2*np.pi, transform=None, pre_transform=None, make_graph=False):
        super().__init__(None, transform, pre_transform)
        
        # get frame_graph dict for each agent in the scene.
        # one item in this dictionary corresponds to one sequence of an agent (with all agents he encounter)
        if make_graph:
            _, self.X = self.make_frame_graphs(dfs)

        # Static nodes, i.e., background sensing.
        # For each node in each frame, we have to let them sense the environment.
        
        self.background_obs_util = BackgroundObservations(background_image_path=background_image_path, homography_mat_path=homography_mat_path, lidar_num_rays=lidar_num_rays, lidar_max_dist=lidar_max_dist, lidar_fov=lidar_fov)
    
    def make_graph(self, frame, with_hidden_state=True, dim_hidden_state=12):
        x = np.arange(len(frame)//2) ## source and target are equal, thats why we divide by 2
        y = np.arange(len(frame)//2)

        # fully connected graph
        edges = np.array([[x0, y0] for x0 in x for y0 in y if x0 != y0]).T
        edge_index = torch.tensor(edges, dtype=torch.long)

        # get the frames
        fs = sorted(np.unique(frame.frame_id))
        x = torch.tensor(frame[frame.frame_id == fs[0]].values, dtype=torch.float)
        y = torch.tensor(frame[frame.frame_id == fs[1]].values, dtype=torch.float) # target values

        if with_hidden_state:
            x = torch.concat( [x, torch.zeros(x.shape[0], dim_hidden_state)], dim=1)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    def make_frame_graphs(self, dfs):

        # Merge dataframe, normalize frames to be incremented by 1
        df = pd.concat(dfs)
        df = df[["agent_id", "frame_id", "pos_x", "pos_y", "vel_x" ,"vel_y", "goal_x", "goal_y"]]
        df.frame_id = (df.frame_id - min(df.frame_id)) // 6

        # do one datapoint for each agents full trajectory
        agents = list(set(df.agent_id))

        # Data for each agent
        X = {}
        for a in agents:
            # get all frame numbers for the agent a
            frames = df[df.agent_id == a].frame_id.values

            # get all rows corresponding to each frame
            df_temp = df[df.frame_id.isin(frames)].copy(deep=True)

            # goal positions
            ## -> last position of each agent.


            # for each frame
            frame_data = []
            for f in frames[:-1]:
                # get all source nodes in current frame
                source = df_temp[(df_temp.frame_id == f)]
                # get all target nodes in next frame
                target = df_temp[(df_temp.frame_id == f+1)]

                # => len of rows must match the number of unique agent ids
                assert(len(set(source.agent_id)) == len(source))
                
                # get all pairs that are in source and target
                target_values = target[target.agent_id.isin(source.agent_id.values)]
                
                # get all source nodes for given target nodes.
                source_values = source[source.agent_id.isin(target_values.agent_id.values)]

                # if source is not in target, 
                # Two options: A) copy source agent row to target (standstill)
                #              B) simply drop it, because at this point the agent will leave and in the training data it does not matter.
                # Lets go with B) for now.
                #source_not_in_target = source[~source.agent_id.isin(target_values.agent_id)].copy(deep=False)

                # concat the two frames again into dataframe
                final_df = pd.concat((source_values, target_values))
                ## sort by frame id first, then sort agents.
                final_df = final_df.sort_values(by=["frame_id", "agent_id"]) 
                frame_data.append(final_df)

            X[a] = frame_data


        #### Make the life and death maps.
        live_death_map = []
        
        for aid in X.keys():
            df_first = X[aid][0]
            df_last = X[aid][-1]
            first = df_first[df_first.agent_id == aid][["frame_id", "pos_x", "pos_y"]].values[0]
            last = df_last[df_last.agent_id == aid][["frame_id", "pos_x", "pos_y"]].values[-1]
            live_death_map.append([first, last])

        life_dict = defaultdict(list)
        death_dict = defaultdict(list)
        for agent_lifes in live_death_map:
            life = agent_lifes[0]
            death = agent_lifes[1]
            life_dict[life[0]].append( life[1:] )
            death_dict[death[0]].append( death[1:] )

        graphs = {}
        for a, x in X.items():
            frms = pd.concat(x).frame_id.values
            min_frame, max_frame = min(frms), max(frms)
            ld_maps = []
            #print(frms)
            for f in range(min_frame, max_frame):
                positions_life = life_dict[f]
                positions_death = death_dict[f]

                life_map = copy.deepcopy(background_obs_utils.occupancy_map)
                life_map[:] = 0
                if len(positions_life) > 0:
                    map_positions = np.flip(background_obs_utils.world2image(np.array(positions_life)), axis=1) ## (y, x)
                    life_map[map_positions[:,0], map_positions[:,1]] = 1

                death_map = copy.deepcopy(background_obs_utils.occupancy_map)
                death_map[:] = 0
                if len(positions_death) > 0:
                    map_positions = np.flip(background_obs_utils.world2image(np.array(positions_death)), axis=1) ## (y, x)
                    death_map[map_positions[:,0], map_positions[:,1]] = 1
                
                ld_maps.append(torch.cat(( torch.Tensor(life_map), torch.Tensor(death_map) ), dim=2))

            #print(len(ld_maps), len(x))
            assert(len(ld_maps) == len(x))
            graphs[a] = [ (make_graph(t), ld_maps[i].permute(2,0,1)) for i, t in enumerate(x)]


        return X, graphs
        

    def make_frame_graphs_legacy(self, dfs):
        df = pd.concat(dfs)
        df = df[["agent_id", "frame_id", "pos_x", "pos_y", "vel_x" ,"vel_y"]]
        df.frame_id = (df.frame_id - min(df.frame_id)) // 6

        #df_filt = df[filt].sort_values(by="frame_id")
        # do one datapoint for each agents full trajectory
        agents = list(set(df.agent_id))

        # data for each agent
        X = {}
        for a in agents:
            frames = df[df.agent_id == a].frame_id.values
            
            ## Forward and backward fill to have the same number of nodes in every frame.
            ## Nodes that did not appear before just stand still in their initial positions
            ## Nodes that leave the scene just stand still at their last position
            df_temp = df[df.frame_id.isin(frames)].copy(deep=True)
            min_frame = df_temp['frame_id'].min()
            max_frame = df_temp['frame_id'].max()
            unique_ids = df_temp['agent_id'].unique()
            new_frame = pd.DataFrame([(frame, id) for frame in range(min_frame, max_frame + 1) for id in unique_ids],
                                     columns=['frame_id', 'agent_id'])
            merged_df = pd.merge(new_frame, df_temp, on=['frame_id', 'agent_id'], how='left')
            merged_df[["pos_x", "pos_y", "vel_x", "vel_y"]] = merged_df.groupby('agent_id')[["pos_x", "pos_y", "vel_x", "vel_y"]].ffill()
            merged_df[["pos_x", "pos_y", "vel_x", "vel_y"]] = merged_df.groupby('agent_id')[["pos_x", "pos_y", "vel_x", "vel_y"]].bfill()

            
            #print(frames)
            frame_data = []
            for f in frames[:-1]:
                ## get all agents and their values in each frame
                source = merged_df[(merged_df.frame_id == f)]
                target = merged_df[(merged_df.frame_id == f+1)]
                ## unique agents in this frame
                ## => len of rows must match the number of unique agent ids

                assert(len(set(source.agent_id)) == len(source))

                ### We need the next row as well

                # get all pairs that are in source and target
                target_values = target[target.agent_id.isin(source.agent_id.values)]

                # if source is not in target, copy source agent row to target
                source_to_target = source[~source.agent_id.isin(target_values.agent_id)].copy(deep=False)
                # only if there is a target not in source
                if len(source_to_target) > 0:
                    # increment frame id to match next frame
                    source_to_target["frame_id"] += 1
                    # concanate the values with target
                    target_values = pd.concat((target_values, source_to_target))
                # concat everything to one dataframe
                final_df = pd.concat((source, target_values))
                ## sort by frame id first, then sort agents.
                final_df = final_df.sort_values(by=["frame_id", "agent_id"]) 
                frame_data.append(final_df)

            X[a] = frame_data

        # create graph object for each frame of each agent
        # graphs[agent_id] = [list of pyg graphs]
        graphs = {}
        for a, x in X.items():
            graphs[a] = [self.make_graph(t) for t in x]

        return graphs
    
    def make_graph_legacy(self, frame, with_hidden_state=True, dim_hidden_state=16-4):
        # node number would be
        #x = np.array(frame.agent_id) 
        #y = np.array(frame.agent_id)

        # actually, the range of nodes is required by pytorch. so we pass the id of the agent as a value of the node.
        # number of nodes are number agents which is number of elements in frame (no duplicates exist)
        x = np.arange(len(frame)//2) ## source and target are equal, thats why we divide by 2
        y = np.arange(len(frame)//2)
        edges = np.array([[x0, y0] for x0 in x for y0 in y if x0 != y0]).T
        edge_index = torch.tensor(edges, dtype=torch.long)
        
        # get the frames
        fs = sorted(np.unique(frame.frame_id))
        x = torch.tensor(frame[frame.frame_id == fs[0]].values, dtype=torch.float)
        y = torch.tensor(frame[frame.frame_id == fs[1]].values, dtype=torch.float) # target values

        # concatenate state space -> fill with 0 values
        # only needed for x, not needed for y
        if with_hidden_state:
            x = torch.concat( [x, torch.zeros(x.shape[0], dim_hidden_state)], dim=1)
            #y = torch.concat( [y, torch.zeros(x.shape[0], dim_hidden_state)-1], dim=1)

        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    def vis_graph(self, data):
        g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        nx.draw(g)

    def draw_trajectory_from_graph(self, gs, cmap="Spectral", figsize=(5,5)):
        plt.figure(figsize=figsize)
        # gs = list of pyg graphs
        positions = [g.x[:, 2:4] for g in gs]

        the_cmap = matplotlib.cm.get_cmap(cmap)
        norm = plt.Normalize(0, len(positions))
        #the_cmap.set_under(0)
        #the_cmap.set_over(len(positions))
        for i, p in enumerate(positions):
            plt.scatter(p[:,0], p[:,1], color=the_cmap(norm(1+i)))

    
    def len(self):
        return len(self.X)
    
    def get(self, idx):
        frames = list(self.X.values())[idx]
        return frames
    

class PreprocessedDataset(Dataset):

    def __init__(self, path):
        self.data_paths = sorted(glob.glob(os.path.join(path, "*")))

    def len(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        return torch.load(self.data_paths[idx])
