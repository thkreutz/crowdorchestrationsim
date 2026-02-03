import os
import pandas as pd
import numpy as np
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


class ETH_Agent_Loader:

    def __init__(self, H_path="/workspace/SceneRepresentation/TKEnvironment/data/ewap_dataset_light/ewap_dataset/seq_eth/H.txt", 
                 agent_data_path="/workspace/SceneRepresentation/TKEnvironment/data/ETH_agents"):
        self.H_path = H_path
        self.agent_data_path = agent_data_path
        self.H = np.loadtxt(H_path)
        self.H_inv = np.linalg.inv(self.H)
        self.agent_data_path = agent_data_path
        self.agents = sorted(glob.glob(os.path.join(self.agent_data_path, "*")))
        self.start_positions, self.end_positions = self.init_start_goal_positions()
        
    def init_start_goal_positions(self):
        positions=[]
        for i in range(len(self.agents)):
            _, pos, _,_ = self.get_agent(index=i)
            positions.append(pos)
           
        start_positions = np.vstack([ [p[0][0], p[1][0]] for p in positions])
        end_positions = np.vstack([ [p[0][-1], p[1][-1]] for p in positions])
        
        return start_positions, end_positions
        
    def world2image(self, traj_w, H_inv):    
        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
        # to camera frame
        traj_cam = np.matmul(H_inv, traj_homog)  
        # to pixel coords
        traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
        return traj_uvz[:, :2].astype(int)    
    
    def get_agent(self, fps=25, index=0):
        
        # Random sampling
        #if seed > 0:
        #    np.random.seed(seed)
        #rand = np.random.randint(len(self.agents))

        traj = pd.read_csv(self.agents[index])
        
        # 25fps = 40ms
        if fps==25:
            traj["datetime"] = pd.to_datetime(traj.timestamp, unit='s')
            traj = traj.set_index('datetime').resample("40L").first().interpolate('time')
        
        traj["goal_x"] = traj["pos_x"].values[-1]
        traj["goal_y"] = traj["pos_y"].values[-1]
        return traj, self.world2image(traj[["pos_x", "pos_y"]].values, self.H_inv).T, traj[["vel_x", "vel_y"]].values.T, traj.frame_id.values
    
    def sample_agent(self, seed=0):
        if seed > 0:
            np.random.seed(seed)
        rand = np.random.randint(len(self.agents))