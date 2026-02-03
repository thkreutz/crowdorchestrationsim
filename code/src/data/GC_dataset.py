import cv2
import pandas as pd
import numpy as np
import torch

from src.transform.GC_Transformer  import GCTransformer


def resize(images, factor, seg_mask=False):
	for key, image in images.items():
		if seg_mask:
			images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
		else:
			images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

def pad(images, division_factor=16):
	""" Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
	at it's bottlenet layer"""
	for key, im in images.items():
		if im.ndim == 3:
			H, W, C = im.shape
		else:
			H, W = im.shape
		H_new = int(np.ceil(H / division_factor) * division_factor)
		W_new = int(np.ceil(W / division_factor) * division_factor)
		im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_REPLICATE)
		images[key] = im

class GC_Dataset():
    
    def __init__(self, k, with_dfs=True):
        self.k = k
        
        self.coord_transforms = {}
        self.bg_images = {}
        self.images_reference = {}
        self.vit_images = {}
        
        self.DFs = {}

        self.img_reference = cv2.imread("OpenTraj/datasets/GC/reference.jpg")
        #self.img_reference = torch.Tensor(self.img_reference).permute(2, 0, 1)
        
        #### Mask annotated with https://www.cvat.ai/
        img_semantic = cv2.imread("OpenTraj/datasets/GC/mask.png", flags=0)
 
        
        #### Make an image consisting of one channel for each semantic class.
        ### TODO[] -> mapping semantic classes for SDD
        # find all classes => one channel for each class
        #for c in np.unique(img_semantic):
        ### SDD Mapping
        # 0 = walkable space
        # 40 = building
        # 174 = obstacle 
        
        channels = []
        #### 5 Channels
        for c in [0, 40, 174]:
            # shape of img
            ch = np.zeros(img_semantic.shape)
            # set 1 where channel is active
            ch[np.where(img_semantic == c)] = 1
            channels.append(ch)

        # stack for image with n_classes as channels
        self.img_semantic = np.transpose(np.dstack(channels), axes=[2,0,1])


        transform = GCTransformer()
        
        self.coord_transforms["gc"] = transform

               
        #self.bg_images["gc"] = self.img_semantic
        self.images_reference["gc"] = self.img_reference
        
        self.bg_images["gc"] = self.img_semantic
        

        ##### Try the reference image instead of the segmentation mask.

        #self.vit_images["gc"] = self.vit_image_preprocessing(k, np.transpose(self.img_reference, axes=[2,0,1]) / 255, seg_mask=False)
        self.vit_images["gc"] = self.vit_image_preprocessing(k, self.img_semantic)

        # Read X and dataframe

        #dddd = [temp[1] for temp in list(df_read.groupby("agent_id"))]
        #self.trajectories = {int(df_agent.agent_id.values[0]) : df_agent for df_agent in dfs}

        if with_dfs:
            self.DFs["gc"] = { aid : df for aid, df in list(pd.read_pickle("data_preprocessed/iclr_data/grand_central/gc_df.pkl").groupby("agent_id")) }
                
                
    def prepare_data(self, dataset, x, i, n_steps=20, dim_hidden_state=16, device="cpu", test=False):
           
        ##### We have the current dataset framenumber in x[i].x
        seq_start_frame = x[0].x[0, 0].int().item()
        current_frame = seq_start_frame + i
        
        # Get the agent id
        aids = x[i].x[:,1]

        # Get the agent's ground truth position
        # State either ONLY x, y (simulation) OR x, y, vx, vy (momentary) 
        #x_state = x[i].x[:, 2:6]  # x, y, vx, vy
        x_state = x[i].x[:, 2:4]  # x, y


        # Add hidden state
        x_state = torch.concat( [x_state, torch.zeros(x_state.shape[0], dim_hidden_state)], dim=1).to(device)
        
        # Target positions
        targets = x[i].y[:, 2:4].to(device)
        
        
        #### Goal is last point of the n_steps sequence. 
        ## -> Goal is to teach the model to MOVE towards this goal by doing imitiation learning/behavior cloning over n_steps
        ## -> By randomly varying the length of the sequence, we teach the model to move towards this point over arbitrary long sequences
        # For the respective agent id, we can get the full sequence like this
        x_goal = []
        for aid in aids:
            # Get the full trajectory of the agent 
            temp = self.DFs[dataset][int(aid.item())]
            
            #print(temp)

            # Get the respective agent's start frame
            agent_start_frame = temp.frame_id.values[0]

            # Get the agent's position
            temp = temp[ ["pos_x", "pos_y"]].values
            #print(temp)
            #print(temp.shape)
            # If the agent has started after the start frame of the current sequence:
            if agent_start_frame - seq_start_frame >= 0:
                # Case 1 and 2 - agent is in the scene since or after start frame

                # do n_steps from start position.
                # if start frame of agent is not at the first sequence step, do n_steps - elapsed_steps
                # => get number of steps that have already elapsed.
                elapsed_steps = agent_start_frame - seq_start_frame 
                
                # Get the frame of the goal.
                # Either: remaining agent steps are less than the remaining steps that the model will run
                #     Or: remaining steps are smaller 
                # => Take one of these positions as goal position.
                g_idx = min(n_steps - elapsed_steps, len(temp)-1)

                #print(g_idx)
                g = temp[int(g_idx)]
            
            # Case 3
            # agent is already in the scene and has started at a previous frame

            else:
                # in this case, we 
                # A) take the difference of the seq_start_frame as an offset and go n_steps
                # or B) take the last position of the agent which will end before this. 
                before_steps = seq_start_frame - agent_start_frame
                
                g_idx = min(before_steps + n_steps, len(temp)-1)

                #print(g_idx)
                g = temp[int(g_idx)]
            
            # Case 4
            # Agent is about to leave the scene
            # => Sample a random goal, reset hidden state, and train him with "common sense loss"
            
            # Add goal
            x_goal.append(g)

        # Tensor from goal
        x_goal = torch.Tensor(np.vstack(x_goal)).to(device)

        # Get the edge index for the graph

        # Modify: -> Edge idx only social neighborhood

        edge_idx = x[i].edge_index.to(device)
        
        # compute heading
        #headings = torch.arctan2(x_state[:,0]-x_goal[:,0], x_state[:,1]-x_goal[:,1])
        #x_distance = (targets[:,:2]-x_goal).pow(2).sum(1).to(device)

        if test:
            # Return the frames as well for testing so that we know exactly when an agent was in the scene.
            return x_state, edge_idx, x_goal, targets, aids, current_frame
        else:
            return x_state, edge_idx, x_goal, targets, aids
    
    def vit_image_preprocessing(self, k, img_semantic, seg_mask=True):
        # Prepare image, we scale the image with a scale factor 1/k and pad to be divisble by patch size.
        scale_factor = 1/k

        im = img_semantic.transpose(1,2,0)
        im_dict = { "k" : im}
        resize(im_dict, scale_factor, seg_mask=seg_mask)
        # dont pad, we dont use patches....
        pad(im_dict, division_factor=16)  # we only try patch sizes 8 and 16 
        
        
        # can we simply scale the positions? yes
        #plt.imshow(im_dict["k"])
        #plt.scatter(pixel_coords[:,0] * scale_factor, pixel_coords[:,1] * scale_factor)
        
        # Put the channel as first.
        img = torch.Tensor(im_dict["k"].transpose(2,0,1))
        return img 

    def get_dataset(self, dataset="gc"):
        return self.coord_transforms[dataset], self.vit_images[dataset], self.DFs[dataset]


    def len(self):
        return len(self.Xs)
    
    

