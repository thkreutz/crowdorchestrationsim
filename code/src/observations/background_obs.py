import cv2
import numpy as np
import copy
import math
import torchvision
import torch

class MapObservations():
     
    def __init__(self,  background_image, 
                        transform, 
                        observation_type="map", 
                        width_crop=300, 
                        height_crop=300,
                        map_padding=300):
        # Background image with white = occupied, black=free

        self.transform = transform
        self.background_image = background_image
        
        #self.occupancy_map = np.zeros((background_image.shape[0], background_image.shape[1]))
        #i_y, i_x = np.where(background_image != 0)
        #self.occupancy_map[i_y, i_x] = 1
        
        self.pad = torch.nn.ZeroPad2d(map_padding)
        self.map_padding = map_padding    # make sure that padding makes the image divisble by 64 in widht and height
        self.background_map = self.pad(torch.Tensor(self.background_image))

        #self.occupancy_map[300:310, 250:350] = 1

        self.y_cells = np.arange(self.background_map.shape[1])
        self.x_cells = np.arange(self.background_map.shape[2])

        self.width_crop = width_crop
        self.height_crop = height_crop

        self.observation_type = observation_type

    
    def circle_mask(self, cx, cy, r):
        mask = (self.x_cells[np.newaxis,:] - cx)**2 + (self.y_cells[:,np.newaxis]-cy)**2 < r**2
        return mask

    def get_observations(self, positions):
        return self.get_map_observations(positions, self.width_crop, self.height_crop)

    def world2image(self, positions):
        ## add padding to the positions to keep them in the right image coords
        return self.transform.get_pixel_positions(positions) + self.map_padding
    
    def image2world(self, positions):
        # remove padding to map back to the right coords
        return self.transform.get_positions(positions - self.map_padding) 

    def world2image_torch(self, positions):
        # clip positions
        #l = torch.tensor([0, 0])
        #u = torch.tensor([self.background_map.shape[1]-1, self.background_map.shape[2]-1])
        #a = torch.tensor([[-1, -1], [10, 6]])
        positions = self.transform.get_pixel_positions_torch(positions) + self.map_padding
        positions[:,0] = torch.clamp(positions[:,0], min=0, max=self.background_map.shape[1]-1)
        positions[:,1] = torch.clamp(positions[:,1], min=0, max=self.background_map.shape[2]-1)
        return positions
    
    def image2world_torch(self, positions):
        return self.transform.get_positions_torch(positions - self.map_padding)
    
    def get_map_observations(self, positions, width, height):
            if type(positions) == torch.Tensor:
                positions = positions.numpy()
            
            
            #map_positions = np.flip(self.world2image(positions), axis=1)  ## (y, x)
            
            map_positions = self.world2image(positions)
            
            #map_positions = map_positions #+ self.map_padding  ### -> translation due to padding
            agent_occupancy_map = np.zeros((self.background_map.shape[1], self.background_map.shape[2]))
            #agent_occupancy_map[:] = 0 # set everything to zero
            
            #print(self.background_map.shape)
            #print(agent_occupancy_map.shape)
            #print(map_positions)
            for p in map_positions:
                agent_occupancy_map[self.circle_mask(p[0], p[1], 3)] = 1

            map = torch.cat( (self.background_map, torch.Tensor(agent_occupancy_map).unsqueeze(0)) )
            #print(map.shape)  ### should add one channel for the agents
            ## Extract local crops for each agent
            agent_obs = []
            for p in map_positions:
                #print(height,width)
                
                crop = torchvision.transforms.functional.crop(map, p[1] - int(height / 2), p[0] - int(width / 2), height, width)
                
                #print(map.shape, (p[1] - int(height / 2), p[0] - int(width / 2)), crop.shape)
                if (crop.shape[1] > self.height_crop) or (crop.shape[2] > self.width_crop):
                    ### if this is the case, then just add an empty crop... I can not figure out why this is happening
                    print("greater crop")
                    ### it might have to with the agent going out of the map. But the torch function should work correctly even in this case...
                    agent_obs.append(torch.zeros((self.background_map.shape[0]+1, self.background_map.shape[1], self.background_map.shape[2])))
                    #torch.save(crop, "why.pt")
                else:
                    agent_obs.append(crop)
                #print(p)
                #print(crop.shape)
                

            ## Returns (2, h, w) local map crops as observations for each agent
            return torch.stack(agent_obs) #, agent_occupancy_map        