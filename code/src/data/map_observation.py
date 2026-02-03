import cv2
import numpy as np
import copy
import math
import torchvision
import torch

class MapObservations():
     
    def __init__(self, background_image_path, 
                        homography_mat_path, 
                        observation_type="map", 
                        width_crop=150, 
                        height_crop=150,
                        map_padding=150):
        # Background image with white = occupied, black=free
        
        #bg = cv2.imread("/home/thomas.kreutz/workspace/SceneRepresentation/TKEnvironment/data/ewap_dataset_full/ewap_dataset/seq_eth/map.png")
        bg = cv2.imread(background_image_path)
        
        self.occupancy_map = np.zeros((bg.shape[0], bg.shape[1]))
        i_y, i_x,_ = np.where(bg == 255)
        self.occupancy_map[i_y, i_x] = 1
        self.pad = torch.nn.ZeroPad2d(map_padding)
        self.map_padding = map_padding
        self.occupancy_map = self.pad(torch.Tensor(self.occupancy_map))

        #self.occupancy_map[300:310, 250:350] = 1

        self.y_cells = np.arange(self.occupancy_map.shape[0])
        self.x_cells = np.arange(self.occupancy_map.shape[1])

        #H = (np.loadtxt("/home/thomas.kreutz/workspace/SceneRepresentation/TKEnvironment/data/ETH/H.txt"))
        self.H = np.loadtxt(homography_mat_path) 
        self.H_inv = np.linalg.inv(self.H) ### Needed to transform the positions to image space
        self.H_inv_torch = torch.Tensor(self.H_inv)
        self.H_torch = torch.Tensor(self.H)

        self.width_crop = width_crop
        self.height_crop = height_crop

        self.observation_type = observation_type

    def world2image(self, traj_w, toint=True):    
        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
        # to camera frame
        traj_cam = np.matmul(self.H_inv, traj_homog)
        # to pixel coords
        traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
        if toint:
            return traj_uvz[:, :2].astype(int) + self.map_padding
        else:
            return traj_uvz[:, :2]  + self.map_padding
    
    def world2image_torch(self, traj_w, toint=True):
        #traj_w = torch.tensor(traj_w)
        #H_inv = torch.tensor(H_inv)

        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = torch.cat((traj_w, torch.ones((traj_w.shape[0], 1))), dim=1).t()
        
        # to camera frame
        traj_cam = torch.matmul(self.H_inv_torch, traj_homog).to(torch.float32)
        
        # to pixel coords
        traj_uvz = traj_cam / traj_cam[2]
        traj_uvz = traj_uvz.t()
        
        if toint:
            return traj_uvz[:, :2].to(torch.int)   + self.map_padding
        else:
            return traj_uvz[:, :2]   + self.map_padding
    
    def image2world_torch(self, traj_uvz):
        traj_uvz = traj_uvz - self.map_padding
        traj_homog = torch.cat((traj_uvz, torch.ones((traj_uvz.shape[0],1)) ), dim=1)
        traj_cam = torch.matmul(self.H_torch, traj_homog.t())
        traj_w = traj_cam / traj_cam[2]
        traj_w = traj_w.t()
        #traj_w = (traj_w[:, :2] / traj_w[:, 2]).reshape(-1,1)
        return traj_w[:, :2] 
        
    
    def image2world(self, traj_uvz):
        traj_uvz = traj_uvz - self.map_padding
        
        # TODO[]: rewrite in pytorch code
 
        # Convert pixel coordinates to homogeneous coordinates
        traj_homog = np.hstack((traj_uvz, np.ones((traj_uvz.shape[0], 1))))
        
        # Apply inverse transformation matrix H
        traj_cam = np.matmul(self.H, np.transpose(traj_homog))
        
        # Convert homogeneous camera coordinates to world coordinates
        traj_w = np.transpose(traj_cam / traj_cam[2])
        
        # Convert homogeneous coordinates to Euclidean coordinates
        traj_w = traj_w[:, :2] / traj_w[:, 2].reshape(-1, 1)
        
        return traj_w #.astype(int)
    
    def circle_mask(self, cx, cy, r):
        mask = (self.x_cells[np.newaxis,:] - cx)**2 + (self.y_cells[:,np.newaxis]-cy)**2 < r**2
        return mask

    def get_observations(self, positions):
        return self.get_map_observations(positions, self.width_crop, self.height_crop)


    def get_map_observations(self, positions, width, height):
            if type(positions) == torch.Tensor:
                positions = positions.numpy()
            map_positions = np.flip(self.world2image(positions), axis=1)  ## (y, x)
            #map_positions = map_positions #+ self.map_padding  ### -> translation due to padding
            agent_occupancy_map = copy.deepcopy(self.occupancy_map)
            agent_occupancy_map[:] = 0 # set everything to zero
            
            
            
            #print(map_positions)
            for p in map_positions:
                agent_occupancy_map[self.circle_mask(p[0], p[1], 3)] = 1

            map = torch.stack( (self.occupancy_map, torch.Tensor(agent_occupancy_map)) )

            ## Extract local crops for each agent
            agent_obs = []
            for p in map_positions:
                #print(height,width)
                
                crop = torchvision.transforms.functional.crop(map, p[1] - int(height / 2), p[0] - int(width / 2), height, width)
                
                #print(map.shape, (p[1] - int(height / 2), p[0] - int(width / 2)), crop.shape)
                if (crop.shape[1] > 150) or (crop.shape[2] > 150):
                    ### if this is the case, then just add an empty crop... I can not figure out why this is happening
                    #print("greater crop")
                    ### it might have to with the agent going out of the map. But the torch function should work correctly even in this case...
                    agent_obs.append(torch.zeros((2, height, width)))
                    torch.save(crop, "why.pt")
                else:
                    agent_obs.append(crop)
                #print(p)
                #print(crop.shape)
                

            ## Returns (2, h, w) local map crops as observations for each agent
            return torch.stack(agent_obs) #, agent_occupancy_map        