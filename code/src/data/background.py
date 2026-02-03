import cv2
import numpy as np
import copy
import math
import torchvision
import torch

class BackgroundObservations():
     
    def __init__(self, background_image_path, homography_mat_path, observation_type="map", width_crop=150, height_crop=150, lidar_num_rays=64, lidar_max_dist=300, lidar_fov=2*np.pi):
        # Background image with white = occupied, black=free
        
        #bg = cv2.imread("/home/thomas.kreutz/workspace/SceneRepresentation/TKEnvironment/data/ewap_dataset_full/ewap_dataset/seq_eth/map.png")
        bg = cv2.imread(background_image_path)
        
        self.occupancy_map = np.zeros((bg.shape[0], bg.shape[1], 1))
        i_y, i_x,_ = np.where(bg == 255)
        self.occupancy_map[i_y, i_x] = 1
        #self.occupancy_map[300:310, 250:350] = 1

        ### Pad the grid to counter small out of bounds scenarios.
        #pad_width = 20
        #self.occupancy_map = np.pad(self.occupancy_map, pad_width, mode='constant')
        
        self.y_cells = np.arange(self.occupancy_map.shape[0])
        self.x_cells = np.arange(self.occupancy_map.shape[1])

        #H = (np.loadtxt("/home/thomas.kreutz/workspace/SceneRepresentation/TKEnvironment/data/ETH/H.txt"))
        self.H = np.loadtxt(homography_mat_path) 
        self.H_inv = np.linalg.inv(self.H) ### Needed to transform the positions to image space -> beam distance will be normalized so no need to transform back

        self.H_inv_torch = torch.Tensor(self.H_inv)

        self.lidar_num_rays = lidar_num_rays
        self.lidar_max_dist = lidar_max_dist
        self.lidar_fov = lidar_fov
        
        self.width_crop = width_crop
        self.height_crop = height_crop

        self.observation_type = observation_type

    def world2image(self, traj_w, H_inv=None):    
        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
        # to camera frame
        traj_cam = np.matmul(self.H_inv, traj_homog)
        # to pixel coords
        traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
        return traj_uvz[:, :2].astype(int)   

    def world2image_torch(self, traj_w):
        #traj_w = torch.tensor(traj_w)
        #H_inv = torch.tensor(H_inv)

        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = torch.cat((traj_w, torch.ones((traj_w.shape[0], 1))), dim=1).t()
        
        # to camera frame
        traj_cam = torch.matmul(self.H_inv_torch, traj_homog)
        
        # to pixel coords
        traj_uvz = traj_cam / traj_cam[2]
        traj_uvz = traj_uvz.t()
        
        return traj_uvz[:, :2].to(torch.int)   
    
    def image2world(self, traj_uvz):
        
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

    
    def cast_rays_parallel(self, occupancy_grid, source_position, id=0, lidar_fov=2*np.pi, lidar_num_rays=128, lidar_max_dist=300):
        y, x = source_position
        angles = np.linspace(0, lidar_fov, lidar_num_rays, endpoint=False)

        targets = np.repeat(np.array( (x, y) ).reshape(1,2), len(angles), axis=0)  ### 
        current_targets = targets

        # -> check for out of bounds
        mask = np.ones(len(current_targets)).astype(bool)
        target_hit_mask = np.ones(len(current_targets)).astype(bool) #((occupancy_grid[current_targets[:,0].astype(int), current_targets[:,1].astype(int)] > 0) & (occupancy_grid[current_targets[:,0].astype(int), current_targets[:,1].astype(int)] != id))[:,0]
        #print(mask, current_targets)
        # Increase depth
        step_size = 1
        for depth in range(lidar_max_dist):
        #depth = 50
            current_targets = current_targets + (((mask & target_hit_mask))[:, np.newaxis] * np.column_stack( (-np.sin(angles) * step_size, np.cos(angles) * step_size)) )
            # out of bounds mask
            mask = (current_targets[:,0] > 0) & (current_targets[:,0] < (occupancy_grid.shape[0]-10)) & (current_targets[:,1] > 0) & (current_targets[:,1] < (occupancy_grid.shape[1]-10))
            # target hit mask
            #print(targets)
            target_hit_mask = ~((occupancy_grid[current_targets[:,0].astype(int), current_targets[:,1].astype(int)] > 0) & (occupancy_grid[current_targets[:,0].astype(int), current_targets[:,1].astype(int)] != id))[:,0]

        ### Finally
        # distance array
        dist_obs = np.sqrt( np.square(targets[:,0] - current_targets[:,0]) + np.square(targets[:,1] - current_targets[:,1]) ) / lidar_max_dist
        #obs = current_targets
        obs = np.column_stack( (current_targets[:,1], current_targets[:,0]) )
        
        return obs, dist_obs
    

    def get_observations(self, positions):
        if self.observation_type == "map":
            return self.get_map_observations(positions, self.width_crop, self.height_crop)
        else:
            return self.get_lidar_observations(positions)


    def get_map_observations(self, positions, width, height):
            map_positions = np.flip(self.world2image(positions, self.H_inv), axis=1) ## (y, x)
            agent_occupancy_map = copy.deepcopy(self.occupancy_map).astype(np.float32)
            agent_occupancy_map[:] = 0 # set everything to zero
            #print(map_positions)
            for p in map_positions:
                agent_occupancy_map[self.circle_mask(p[0], p[1], 3)] = 1

            map = torch.stack( (torch.Tensor(self.occupancy_map.astype(np.float32).squeeze()), torch.Tensor(agent_occupancy_map.squeeze())) )

            ## Extract local crops for each agent
            agent_obs = []
            for p in map_positions:
                crop = torchvision.transforms.functional.crop(map, p[1] - int(height / 2), p[0] - int(width / 2), height, width)
                agent_obs.append(crop)

            ## Returns (2, h, w) local map crops as observations for each agent
            return agent_obs #, agent_occupancy_map        

        


    def get_lidar_observations(self, positions):
        #positions = data[0].x[:, 2:4] ## positions in (x,y)

        # transform to image coordinates and flip positions to be in "image space", i.e., first axis = y, second axis = x
        map_positions = np.flip(self.world2image(positions, self.H_inv), axis=1) ## (y, x)

        occupancy_map = copy.deepcopy(self.occupancy_map)


        #### Make a mask for each particle on the occupancy map and init occupancy map, 
        # 1 means occupied, so we start with id=2 to be able to track particles in the map
        i = 2
        for p in map_positions:
            occupancy_map[self.circle_mask(p[0], p[1], 2)] = i
            i+=1

        ## Raycasting in 360 degrees -> No specified orientation, we are a "surroundings-aware particle connected to all other particles".

        observations = []
        for idd, p in enumerate(map_positions):
            # check for out of bounds of position... invalid
            if (p[0] < 0) or (p[1] >= self.occupancy_map.shape[0]-5) or (p[1] <= 0) or (p[0] >= self.occupancy_map.shape[1]-5):
                dist_obs = np.zeros(self.lidar_num_rays)
            else:
                _, dist_obs = self.cast_rays(occupancy_grid=occupancy_map, source_position=p, id=idd+2, lidar_fov=self.lidar_fov, lidar_num_rays=self.lidar_num_rays, lidar_max_dist=self.lidar_max_dist)
            # dist_obs includes normalized distance for each beam -> 1 is max distance or out of range.
            observations.append(dist_obs)
        
        ## observations is shape (N,lidar_num_rays) -> observation vector with dimension equal to the number of beams.
        ## return occupancy map just to have it...
        return np.array(observations) #, occupancy_map ## could also return the occupancy_map
        
    
    def cast_rays(self, occupancy_grid, source_position, id=0, lidar_fov=2*np.pi, lidar_num_rays=128, lidar_max_dist=300):
        y, x = source_position
        
        # field of view -> 2pi = 360 degree
        angles = np.linspace(0, lidar_fov, lidar_num_rays)

        # store the point where an obstacle was hit for each ray
        obs = []
        dist_obs = []
        # for each ray
        for angle in angles:
            # maximum depth
            for depth in range(lidar_max_dist):
                hit = False
                
                target_x = x - math.sin(angle) * depth
                target_y = y + math.cos(angle) * depth
                if target_x >= occupancy_grid.shape[0]-1 or target_x < 0:
                    break
                if target_y >= occupancy_grid.shape[1]-1 or target_y < 0:
                    break

                col = int(target_x)
                row = int(target_y)

                if (occupancy_grid[col][row] > 0) and (occupancy_grid[col][row] != id):
                    #print("hit")
                    obs.append((target_y, target_x))
                    # compute respective distance
                    hit = True
                    dist_obs.append(np.sqrt((x - target_x)**2  +  (y - target_y)**2) / lidar_max_dist)
                    break
                    # add win as parameter if we still draw line here
                    #pygame.draw.line(win, (255, 255, 0), (self.x, self.y), (target_x, target_y), 2)
            # max distance reached and nothing was found.
            if not hit:
                dist_obs.append(1.0) ### add LiDAR max dist

        return np.array(obs), np.array(dist_obs)

    #def plot_rays(occupancy_map, obs, dist_obs, source_position):
        
        #can use this as a guidance for visualization of the rays and the map...
        #### PLot
        #for j, o in enumerate(obs):
        #    y = [source_position[1], o[1]]
        #    x = [source_position[0], o[0]]
        #    plt.plot(x, y, color=plt.get_cmap("cool")(dist_obs[dist_obs < 1][j]))#

        #y,x,_ = np.where(occupancy_map > 0)
        #plt.scatter(x, y, s=1)
        

