from src.data.loaders.loader_eth import load_eth
from src.data.loaders.loader_gcs import load_gcs
from src.data.loaders.loader_sdd import load_sdd, load_sdd_dir
from src.data.loaders.loader_crowds import load_crowds

from src.data.loaders.loader_edinburgh import load_edinburgh

import pickle
#from src.models.ViT import resize, pad

from src.transform.SDD_Transformer import SDDTransformer
from src.transform.ETH_Transformer import ETHTransformer
from src.transform.UCY_Transformer import UCYTransformer
from src.transform.GC_Transformer  import GCTransformer

from src.observations.background_obs import MapObservations

import torch

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, yaml
import warnings
import pandas as pd
import cv2
import yaml
from tqdm import tqdm
import copy
warnings.filterwarnings('ignore')

#from sequence_dataset import SceneSequenceDataset
#from map_observation import MapObservations

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

#### Using the methods from opentraj, we can read eth, gc, sdd and get the unique trajectories of each agent.
def vis_dataset(traj_dataset, n_trajs=50):
    trajs = traj_dataset.get_trajectories()
    dfs = list(trajs)
    
    for df in dfs[:n_trajs]:
        tra = df[1][ ["pos_x", "pos_y" ]].values
        plt.scatter(tra[:,0], tra[:,1])


def open_grand_central(path="OpenTraj/datasets/GC/Annotation"):
    #### Grand Central Dataset
    path = os.path.join(path)
    traj_dataset = load_gcs(path)

    return traj_dataset

def open_forum(path="OpenTraj/datasets/Edinburgh/annotations", day="01Aug"):
    #### Grand Central Dataset
    path = os.path.join(path, 'tracks.%s.txt' % day)
    traj_dataset = load_edinburgh(path, title="Edinburgh", use_kalman=False, scene_id=day, sampling_rate=4)

    return traj_dataset

def open_atc(path="OpenTraj/datasets/ATC/atc.pkl"):
    # Load the dataframe...
    raw_dataset = pd.read_pickle(path)

    # The positions are way too big, we scale them down.
    raw_dataset["pos_x"] = raw_dataset["pos_x"] / 100
    raw_dataset["pos_y"] = raw_dataset["pos_y"] / 100

    # Also let them start above 0
    raw_dataset["pos_x"] = raw_dataset["pos_x"] - min(raw_dataset["pos_x"]) + 32
    raw_dataset["pos_y"] = raw_dataset["pos_y"] - min(raw_dataset["pos_y"]) + 32

    return raw_dataset
    


def open_eth(name="hotel", eth_path="OpenTraj/datasets/ETH/"):
    assert ((name == "hotel") or (name == "eth")), "hotel or eth expected"
    path = os.path.join(eth_path, "seq_%s/obsmat.txt" % name)
    traj_dataset = load_eth(os.path.join("", path))
    return traj_dataset


def open_sdd(scene_name = 'hyang', scene_video_id = 'video1', sdd_root="OpenTraj/datasets/SDD"):
    # SDD
    #scene_name = 'hyang'
    #scene_video_id = 'video6'

    assert scene_name in ["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"], "no valid scene name"

    # fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
    #sdd_root = os.path.join("../OpenTraj/", 'datasets', 'SDD')
    annot_file = os.path.join(sdd_root, scene_name, scene_video_id, 'annotations.txt')

    # load the homography values
    with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
        scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
    scale = scales_yaml_content[scene_name][scene_video_id]['scale']

    traj_dataset = load_sdd(annot_file, scale=scale, scene_id=scene_name + '-' + scene_video_id,
                            drop_lost_frames=False, use_kalman=False) 
    
    #trajs = traj_dataset.get_trajectories()
    #dfs = list(trajs)

    return traj_dataset


def open_ucy(name="zara1"):
    ucy_annot = os.path.join("OpenTraj/datasets/UCY/%s/annotation.vsp" % name)
    
    
    ucy_H_file = os.path.join("OpenTraj/datasets/UCY/%s/H.txt" % name)
    #ucy_H_file = "ynet/data/eth_ucy/%s_H.txt" % name
    
    
    traj_dataset = load_crowds(ucy_annot, use_kalman=False, homog_file=ucy_H_file)
    return traj_dataset


### based on https://github.com/HarshayuGirase/Human-Path-Prediction/blob/master/ynet/utils/preprocessing.py
### Downsample to 1 fps
def mask_step(x, step):
    """
    Create a mask to only contain the step-th element starting from the first element. Used to downsample
    """
    mask = np.zeros_like(x)
    mask[::step] = 1
    return mask.astype(bool)


def downsample(df, step):
    """
    Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
    df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
    pedestrian (metaId)
    :param df: pandas DataFrame - necessary to have column 'metaId'
    :param step: int - step size, similar to slicing-step param as in array[start:end:step]
    :return: pd.df - downsampled
    """
    mask = df.transform(mask_step, step=step)
    return df[mask].dropna()




class GC_Dataset():
    
    def __init__(self, k):
        self.k = k
        
        self.coord_transforms = {}
        self.bg_images = {}
        self.images_reference = {}
        self.vit_images = {}
        
        self.DFs = {}

        self.img_reference = cv2.imread("OpenTraj/datasets/GC/reference.jpg")
        
        
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
        self.vit_images["gc"] = self.vit_image_preprocessing(k, self.img_semantic)
        # Read X and dataframe

        #dddd = [temp[1] for temp in list(df_read.groupby("agent_id"))]
        #self.trajectories = {int(df_agent.agent_id.values[0]) : df_agent for df_agent in dfs}

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
    
    def vit_image_preprocessing(self, k, img_semantic):
        # Prepare image, we scale the image with a scale factor 1/k and pad to be divisble by patch size.
        scale_factor = 1/k

        im = img_semantic.transpose(1,2,0)
        im_dict = { "k" : im}
        resize(im_dict, scale_factor, seg_mask=True)
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
    
    


class SDD_Dataset():
    def __init__(self, datasets_train, datasets_test, k):
        self.k = k
        self.datasets_train = datasets_train
        self.datasets_test = datasets_test
        
        self.coord_transforms = {}
        self.bg_images = {}
        self.images_reference = {}
        self.vit_images = {}
        self.DFs = {}
        self.Xs = {}
        import yaml
        
        with open("OpenTraj/datasets/SDD/estimated_scales.yaml", 'r') as stream:
            estimated_scales = yaml.safe_load(stream)
            
        for dataset in datasets_train + datasets_test:
            
            scene, video = dataset
            print(scene,video)
            # Read transforms and bg image
            ### Can directly run vit image preprocessing here
            if dataset in datasets_train:
                img_semantic = cv2.imread("ynet/data/SDD_semantic_maps/train_masks/%s_%s_mask.png" % (scene, video), flags=0)
                img_reference = cv2.imread("ynet/data/SDD/train/%s_%s/reference.jpg" % (scene, video), flags=0)
            else:
                img_semantic = cv2.imread("ynet/data/SDD_semantic_maps/test_masks/%s_%s_mask.png" % (scene, video), flags=0)
                img_reference = cv2.imread("ynet/data/SDD/test/%s_%s/reference.jpg" % (scene, video), flags=0)
            self.images_reference[dataset] = img_reference
            
            
            channels = []
            #### Make an image consisting of one channel for each semantic class.
            ### TODO[] -> mapping semantic classes for SDD
            # find all classes => one channel for each class
            #for c in np.unique(img_semantic):
            ### SDD Mapping
            # 1 = road
            # 2 = pavement
            # 3 = structure
            # 4 = terrain
            # 5 = tree

            use_all_channels = True
            if use_all_channels:
                #### 5 Channels
                for c in range(5):
                    # shape of img
                    ch = np.zeros(img_semantic.shape)
                    # set 1 where channel is active
                    ch[np.where(img_semantic == c+1)] = 1
                    channels.append(ch)
            else:

                #### 2 Channels Only
                for c in range(2):
                    # shape of img
                    ch = np.zeros(img_semantic.shape)
                    # set 1 where channel is active
                    if c == 0:
                        ch[np.where(img_semantic == 1)] = 1
                        ch[np.where(img_semantic == 2)] = 1
                        ch[np.where(img_semantic == 4)] = 1
                    if c == 1:
                        ch[np.where(img_semantic == 3)] = 1
                        ch[np.where(img_semantic == 5)] = 1

                    channels.append(ch)

            # stack for image with n_classes as channels
            img_semantic = np.transpose(np.dstack(channels), axes=[2,0,1])


            sdd_scale = estimated_scales[scene]["video%s"%video]["scale"]
            transform = SDDTransformer(sdd_scale)
            
            self.coord_transforms[dataset] = transform

            self.bg_images[dataset] = img_semantic
            self.vit_images[dataset] = self.vit_image_preprocessing(k, img_semantic)
            # Read X and dataframe

            #dddd = [temp[1] for temp in list(df_read.groupby("agent_id"))]
            #self.trajectories = {int(df_agent.agent_id.values[0]) : df_agent for df_agent in dfs}

            self.DFs[dataset] = { aid : df for aid, df in list(pd.read_pickle("data_preprocessed/iclr_data/sdd_only_moving_start/%s_video%s_df.pkl" % dataset).groupby("agent_id")) }
            
            
            with open("data_preprocessed/iclr_data/sdd_only_moving_start/%s_video%s_X.pkl" % dataset, "rb") as f:
                self.Xs[dataset] = pickle.load(f)
                
                
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
    
    def vit_image_preprocessing(self, k, img_semantic):
        # Prepare image, we scale the image with a scale factor 1/k and pad to be divisble by patch size.
        scale_factor = 1/k

        im = img_semantic.transpose(1,2,0)
        im_dict = { "k" : im}
        resize(im_dict, scale_factor, seg_mask=True)
        # dont pad, we dont use patches....
        pad(im_dict, division_factor=32)  # we only try patch sizes 8 and 16 
        
        
        # can we simply scale the positions? yes
        #plt.imshow(im_dict["k"])
        #plt.scatter(pixel_coords[:,0] * scale_factor, pixel_coords[:,1] * scale_factor)
        
        # Put the channel as first.
        img = torch.Tensor(im_dict["k"].transpose(2,0,1))
        return img 

    def get_dataset(self, dataset):
        return self.coord_transforms[dataset], self.vit_images[dataset], self.DFs[dataset], self.Xs[dataset]


    def len(self):
        return len(self.Xs)


class ETH_UCY_Dataset():
    def __init__(self, datasets_train, datasets_test, k):
        self.k = k
        self.datasets_train = datasets_train
        self.datasets_test = datasets_test
        self.coord_transforms = {}
        self.bg_images = {}
        self.vit_images = {}
        self.DFs = {}
        self.Xs = {}
        
        for dataset in datasets_train + datasets_test:

            # Read transforms and bg image
            ### Can directly run vit image preprocessing here
            img_semantic = cv2.imread("ynet/data/eth_ucy/%s/oracle.png" % dataset, flags=0)
            
            channels = []
            #### Make an image consisting of one channel for each semantic class.
            ### TODO[] -> mapping semantic classes for SDD
            # find all classes => one channel for each class
            for c in np.unique(img_semantic):
                # shape of img
                ch = np.zeros(img_semantic.shape)
                # set 1 where channel is active
                ch[np.where(img_semantic == c)] = 1
                channels.append(ch)
            # stack for image with n_classes as channels
            img_semantic = np.transpose(np.dstack(channels), axes=[2,0,1])


            transform = None
            if dataset in ["eth", "hotel"]:
                homography_mat_path = "OpenTraj/datasets/ETH/seq_%s/H.txt" % dataset
                transform = ETHTransformer(homography_mat_path)
            else:
                homography_mat_path = "OpenTraj/datasets/UCY/%s/H.txt" % dataset
                transform = UCYTransformer(homography_mat_path)

            self.coord_transforms[dataset] = transform


            self.bg_images[dataset] = img_semantic
            self.vit_images[dataset] = self.vit_image_preprocessing(k, img_semantic)
            # Read X and dataframe

            #dddd = [temp[1] for temp in list(df_read.groupby("agent_id"))]
            #self.trajectories = {int(df_agent.agent_id.values[0]) : df_agent for df_agent in dfs}

            self.DFs[dataset] = { aid : df for aid, df in list(pd.read_pickle("data_preprocessed/iclr_data/%s_df.pkl" % dataset).groupby("agent_id")) }
            with open("data_preprocessed/iclr_data/%s_X.pkl" % dataset, "rb") as f:
                self.Xs[dataset] = pickle.load(f)

    def prepare_data(self, dataset, x, i, n_steps=20, dim_hidden_state=16, device="cpu", test=False):
       
        ##### We have the current dataset framenumber in x[i].x
        seq_start_frame = x[0].x[0, 0].int().item()
        current_frame = seq_start_frame + i
        
        # Get the agent id
        aids = x[i].x[:,1]
        # Get the agent's ground truth position
        x_state = x[i].x[:, 2:4]
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
            
            # Get the respective agent's start frame
            agent_start_frame = temp.frame_id.values[0]

            # Get the agent's position
            temp = temp[ ["pos_x", "pos_y"]].values

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
                g = temp[g_idx]
            
            # Case 3
            # agent is already in the scene and has started at a previous frame

            else:
                # in this case, we 
                # A) take the difference of the seq_start_frame as an offset and go n_steps
                # or B) take the last position of the agent which will end before this. 
                before_steps = seq_start_frame - agent_start_frame
                
                g_idx = min(before_steps + n_steps, len(temp)-1)

                #print(g_idx)
                g = temp[g_idx]
            
            # Case 4
            # Agent is about to leave the scene
            # => Sample a random goal, reset hidden state, and train him with "common sense loss"
            
            # Add goal
            x_goal.append(g)

        # Tensor from goal
        x_goal = torch.Tensor(np.vstack(x_goal)).to(device)

        # Get the edge index for the graph
        edge_idx = x[i].edge_index.to(device)
        
        # compute heading
        #headings = torch.arctan2(x_state[:,0]-x_goal[:,0], x_state[:,1]-x_goal[:,1])
        #x_distance = (targets[:,:2]-x_goal).pow(2).sum(1).to(device)

        if test:
            # Return the frames as well for testing so that we know exactly when an agent was in the scene.
            return x_state, edge_idx, x_goal, targets, aids, current_frame
        else:
            return x_state, edge_idx, x_goal, targets, aids
    
    def vit_image_preprocessing(self, k, img_semantic):
        # Prepare image, we scale the image with a scale factor 1/k and pad to be divisble by patch size.
        scale_factor = 1/k

        im = img_semantic.transpose(1,2,0)
        im_dict = { "k" : im}
        resize(im_dict, scale_factor, seg_mask=True)
        pad(im_dict, division_factor=16)  # we only try patch sizes 8 and 16 
        # can we simply scale the positions? yes
        #plt.imshow(im_dict["k"])
        #plt.scatter(pixel_coords[:,0] * scale_factor, pixel_coords[:,1] * scale_factor)
        
        # Put the channel as first.
        img = torch.Tensor(im_dict["k"].transpose(2,0,1))
        return img 

    def get_dataset(self, dataset):
        return self.coord_transforms[dataset], self.vit_images[dataset], self.DFs[dataset], self.Xs[dataset]


    def len(self):
        return len(self.Xs)



def clean_sequences_data_preprocessing(DFs, filter_length=False):

    clean_dfs = []
    for df in DFs:
        xx = df[["pos_x", "pos_y"]].values

        # Get the difference between goal and start position to filter out agents that only stand still.
        diff = (df.goal_x.values[0] - df.pos_x.values[0])**2 + (df.goal_y.values[0] - df.pos_y.values[0])**2
        
        # just make based on velocity?

        #if filter_length:
        #    if (diff > 2) and (len(df) < 100): # want that agents moves a good amount of space and no ultra long trajectories
        
        clean_dfs.append(df)

    print("n_before=%s, n_after=%s" % (len(DFs), len(clean_dfs)))

    return clean_dfs   


class DatasetLoader():
    
    def __init__(self, dataset="sdd", sdd_scene_name = "bookstore", sdd_seq=0, dim_hidden_state=12, min_length_seqs=20, with_hidden_state=False, with_graph_sequences=False):
        
        self.dim_hidden_state = dim_hidden_state
        
        self.dataset = dataset
        self.sdd_scene_name = sdd_scene_name
        self.sdd_seq = sdd_seq
        self.min_length_seqs = min_length_seqs
        
        self.with_graph_sequences = with_graph_sequences
        self.with_hidden_state = with_hidden_state
        ### Paths are hardcoded, paths can be made variable
        #self.load()
        #self.preprocess()

    def visualize_agents(self):
        fig, ax = plt.subplots(1,2, figsize=(20,10))


        ax[0].imshow(self.img)
        ax[1].imshow(self.img_semantic)

        for i in range(len(self.dfs))[:200]:
            
            test = self.transform.get_pixel_positions(self.dfs[i][["pos_x", "pos_y"]].values)
            ax[0].scatter(test[:,0], test[:,1] )

    def load(self):
        
        if self.dataset == "sdd":
            sdd_scene_video_id = 'video%s'% self.sdd_seq
            sdd_scene = "%s_%s" % (self.sdd_scene_name, self.sdd_seq)
            
            traj_dataset = open_sdd(self.sdd_scene_name, scene_video_id = sdd_scene_video_id)
            

        elif self.dataset == "gc":

            traj_dataset = open_grand_central()
            
        elif self.dataset == "hotel":
            traj_dataset = open_eth(name="hotel")
 
        elif self.dataset == "eth":
            traj_dataset = open_eth(name="eth")


        elif self.dataset.startswith("zara"):

            traj_dataset = open_ucy(name=self.dataset)

        elif self.dataset.startswith("students"):
            print("hi")
            traj_dataset = open_ucy(name=self.dataset)


        elif self.dataset.startswith("forum"):
            traj_dataset = open_forum(day=self.sdd_scene_name)
        
        elif self.dataset.startswith("atc"):
            df = open_atc()

        else:
            print("expected sdd, gc, eth_hotel, or eth")
        

        if not self.dataset.startswith("atc"):
            dfs = traj_dataset.get_trajectories()
            dfs = [d[1] for d in list(dfs) if len(d[1]) > self.min_length_seqs] ### We only keep trajectories that are at least 20 timesteps long.
        else:
            dfs = list(df.groupby("agent_id"))
            dfs = [d[1] for d in list(dfs) if len(d[1]) > 50 and len(d[1]) < 2500] ### checked histogram...
            dfs = [downsample(df, 15) for df in dfs] # downsample because resolution is way too high...

        
        if self.dataset == "sdd" :
            # only keep pedestrians
            dfs = [df for df in dfs if all(df.label == "pedestrian")] 
            # only keep non-disturbed paths
            
            # downsample to 1fps (based on "from waypoints to goals to paths")
            # step=30 means 1fps and step=12 means 2.5fps on SDD
            dfs = [downsample(df, 30) for df in dfs]


        #### make goal and start to be the last position of the sequence , we normalize by the starting position later on.
        dfs = [self.add_goal(df) for df in dfs]


        #### Clean trajectories where people are just standing still
        dfs = clean_sequences_data_preprocessing(dfs)

        ## Init variables
        self.dfs = dfs

        #self.trajectories = {int(df_agent.agent_id.values[0]) : df_agent for df_agent in dfs}
        
        
    def add_goal(self, df):
        ### We treat last position of each agent as is its goal, make new column in each dataframe for that
        df["goal_x"] = df["pos_x"].values[-1]
        df["goal_y"] = df["pos_y"].values[-1]
        df["start_x"] = df["pos_x"].values[0]
        df["start_y"] = df["pos_y"].values[0]
        return df
    
    
    def preprocess(self):
        if self.dataset.startswith("zara"):
            dataset = "zara"
        if self.dataset.startswith("students"):
            dataset = "students"
        else:
            dataset = self.dataset
            
        
        ### Diff, so that frames get incremented by 1.
        diffs = {   "eth" : 6,
                    "hotel" : 10,
                    "sdd" : 30, # we sample to 1fps...
                    "gc" : 10,
                    "zara" : 10,
                    "students" : 1,
                    "forum" : 4,
                    "atc" : 30}  

        df = pd.concat(self.dfs)

        # only keep these columns
        #df = df[["frame_id", "agent_id", "pos_x", "pos_y", "vel_x" ,"vel_y", "goal_x", "goal_y", "start_x", "start_y"]]   
        df = df[["frame_id", "agent_id", "pos_x", "pos_y", "goal_x", "goal_y", "start_x", "start_y"]]  

        df.frame_id = (df.frame_id - min(df.frame_id)) // diffs[dataset]
        
        
        ### For each agent we need its spawn and death frame.
        # Get dictionary that holds the first and last position of each agent.
        if False:
            self.agent_spawns = {}
            self.agent_deaths = {}
            for aid in np.unique(df.agent_id.values):
                self.agent_spawns[aid] = list(df[df.agent_id == aid][["frame_id", "pos_x", "pos_y", "vel_x", "vel_y"]].sort_values(by=["frame_id"]).values[0]) # first frame and position
                self.agent_deaths[aid] = list(df[df.agent_id == aid][["frame_id", "pos_x", "pos_y", "vel_x", "vel_y"]].sort_values(by=["frame_id"]).values[-1]) # last frame and position

            self.agents = sorted(list(self.agent_spawns.keys()))
        

        if self.with_graph_sequences:
            X = self.make_sequences(df, self.agents, self.agent_spawns, self.agent_deaths)
            self.X = list(X.values()) ### just keep the values so that we dont query by keys.
            self.X_dict = X

        self.agent_df = df
    
    def make_graph(self, frame, with_hidden_state=False, dim_hidden_state=12):
        fs = sorted(np.unique(frame.frame_id))
        #print(fs)
            
        x = np.arange(len(frame[frame.frame_id == fs[0]].values)) ## source and target are equal for edge index.
        y = np.arange(len(frame[frame.frame_id == fs[0]].values))

        # fully connected graph
        edges = np.array([[x0, y0] for x0 in x for y0 in y if x0 != y0]).T
        edge_index = torch.tensor(edges, dtype=torch.long)

        # get the frames

        x = torch.tensor(frame[frame.frame_id == fs[0]].values, dtype=torch.float)
        if len(fs) > 1:
            y = torch.tensor(frame[frame.frame_id == fs[1]].values, dtype=torch.float) # target values
        else:
            y = torch.tensor([])

        if with_hidden_state:
            x = torch.concat( [x, torch.zeros(x.shape[0], dim_hidden_state)], dim=1)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    

    def make_sequences(self, df, agents, agent_spawns, agent_deaths):
        X = {}
        print("Creating graph sequences...")
        for aid in tqdm(agents):
            #print(aid)
            graphs = []
            #print(seq)
            start_frame = int(agent_spawns[aid][0])
            end_frame = int(agent_deaths[aid][0])
            #print(start_frame, end_frame)
            seq = df[ (df.frame_id >= start_frame) & (df.frame_id <= end_frame)]
            for f in range(start_frame, end_frame-1):  ## go until the last frame, so that sequence always ends  with death of the current agent that governs sequence
                # get all source nodes in current frame
                source = seq[(seq.frame_id == f)]
                # get all target nodes in next frame
                target = seq[(seq.frame_id == f+1)]
                
                #print(source.agent_id, target.agent_id)
                # Make sure to only keep agents that are present in both source and target.
                source = source[source.agent_id.isin(target.agent_id.values)] ### Remove agents in source that are not in target.
                target = target[target.agent_id.isin(source.agent_id.values)] ### Remove agents in target that are not in source.
                #print(source.agent_id, target.agent_id)
                
                
                final_df = pd.concat([source,target])
                final_df = final_df.sort_values(by=["frame_id", "agent_id"]) 
                
                
                #print(f)
                data = self.make_graph(final_df, with_hidden_state=self.with_hidden_state, dim_hidden_state=12)
                graphs.append(data)
            X[aid] = graphs
        
        return X
    
    
    
