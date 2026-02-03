### For each respective scene, read trajectories, images, and create the coordinate transform
from src.transform.SDD_Transformer import SDDTransformer
from src.transform.ETH_Transformer import ETHTransformer
from src.transform.UCY_Transformer import UCYTransformer
from src.transform.GC_Transformer  import GCTransformer
from src.transform.Forum_Transformer  import ForumTransformer
import os, yaml
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class TrajScene:
    
    def __init__(self, dataset_name, scene_name, scale_factor, data_root):

        min_agent_id = 10
        if dataset_name == "sdd":
            # format scene_name=(scene, video), e.g., ("hyang", 4)
            scene = scene_name[0]
            video = scene_name[1]
            
            with open( os.path.join(data_root, "OpenTraj/datasets/SDD/estimated_scales.yaml"), 'r') as stream:
                estimated_scales = yaml.safe_load(stream)
            self.img_semantic = cv2.imread(os.path.join(data_root, "OpenTraj/datasets/SDD/ynet/data/SDD_semantic_maps/all_masks/%s_%s_mask.png" % (scene, video)), flags=0)
            self.img_reference = cv2.imread(os.path.join(data_root, "OpenTraj/datasets/SDD/ynet/data/SDD/train/%s_%s/reference.jpg" % (scene, video)))
            sdd_scale = estimated_scales[scene]["video%s"%video]["scale"]
            self.coord_transform = SDDTransformer(sdd_scale)
            self.trajs_df = pd.read_pickle("../data_preprocessed/SDD/%s_video%s_df.pkl" % (scene, video) )
            self.trajectories = { aid + min_agent_id: df for aid, df in list(self.trajs_df.groupby("agent_id")) }

            
        elif dataset_name == "eth":
            
            self.img_semantic = cv2.imread(os.path.join(data_root, "OpenTraj/datasets/ETH/seq_%s/oracle.png" % scene_name), flags=0)
            self.img_reference = cv2.imread(os.path.join(data_root, "OpenTraj/datasets/ETH/seq_%s/reference.png" % scene_name))
            
            homography_mat_path = os.path.join(data_root, "OpenTraj/datasets/ETH/seq_%s/H.txt" % scene_name)
            self.coord_transform = ETHTransformer(homography_mat_path)
            
            self.trajs_df = pd.read_pickle("../data_preprocessed/ETH/%s_df.pkl" % (scene_name) )
            self.trajectories = { aid + min_agent_id : df for aid, df in list(self.trajs_df.groupby("agent_id")) }
            
        elif dataset_name == "gc":
            self.img_reference = cv2.imread( os.path.join(data_root, "OpenTraj/datasets/GC/reference.jpg"))
            #### Mask annotated with https://www.cvat.ai/
            self.img_semantic = cv2.imread( os.path.join(data_root, "OpenTraj/datasets/GC/mask.png"), flags=0)
            self.coord_transform = GCTransformer()

            self.trajs_df = pd.read_pickle("../data_preprocessed/GC/%s_df.pkl" % (scene_name)) 
            
            self.trajectories = {aid + min_agent_id : df for aid, df in list(self.trajs_df.groupby("agent_id")) }
        
        elif dataset_name == "forum":
            self.img_reference = cv2.imread( os.path.join(data_root, "OpenTraj/datasets/Edinburgh/reference.jpg"))

            self.img_semantic = cv2.imread( os.path.join(data_root, "OpenTraj/datasets/Edinburgh/mask.png"), flags=0)
            self.coord_transform = ForumTransformer()

            if scene_name == "":
                self.trajs_df = pd.read_pickle("../data_preprocessed/Forum/forum_df.pkl") 
            else:
                self.trajs_df = pd.read_pickle("../data_preprocessed/Forum/forum_%s_df.pkl" % scene_name) 
            
            self.trajectories = {aid + min_agent_id : df for aid, df in list(self.trajs_df.groupby("agent_id")) }

        elif dataset_name == "atc":

            #self.img_reference = cv2.imread("OpenTraj/datasets/ATC/reference.png")
            self.img_reference = cv2.imread("OpenTraj/datasets/ATC/mask.png")

            #### TODO annotate mask

            self.img_semantic = cv2.imread("OpenTraj/datasets/ATC/mask.png", flags=0)

            self.coord_transform = ForumTransformer() # just a transformer that does nothing..

            self.trajs_df = pd.read_pickle("../data_preprocessed/ATC/atc_df.pkl") 
            
            self.trajectories = {aid + min_agent_id : df for aid, df in list(self.trajs_df.groupby("agent_id")) }


        elif dataset_name == "ind":
            self.trajs_df = None
            print("not implemented yet")

        else:
            self.trajs_df = None
            print("not supported") 
        
        ### Get all neighbors in the scene for each agent for crowd replay
        #elf.crowd_trajectories = {}
        #for aid in self.trajectories.keys():
        #    sub_df = self.trajs_df[self.trajs_df.frame_id.isin(self.trajs_df[self.trajs_df.agent_id == aid].frame_id.values)]
        #    sub_trajs = {acid : df for acid, df in list(sub_df.groupby("agent_id")) }
        #    self.crowd_trajectories[aid] = sub_trajs
        #    ### aid can be recovered from within sub_trajs using the same key aid, all other agents are a different id
        #    ### When we query an agent, take the start frame

        ### Different way is to simply store the whole traj data and at each frame we simply query to get all agents that are also in the scene at the same time...
       
        # Set all agent ids to be min at 10 so that we have some values that we can set in the obstacle map without problems.
        self.trajs_df["agent_id"] += min_agent_id
        #for k, v in self.trajectories.items():
        #    # print(k, k+6, name_dict.__sizeof__())
        #    self.trajectories[k + 10] = self.trajectories.pop(k)


        ### [] TODO image scaling, but thats the same for all datasets.
        self.img_semantic_scaled = self.image_scale_pad(scale_factor, self.img_semantic, seg_mask=True)
        self.img_reference_scaled = self.image_scale_pad(scale_factor, self.img_reference, seg_mask=False)
    
    def resize(self, images, factor, seg_mask=False):
        for key, image in images.items():
            if seg_mask:
                images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
            else:
                images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    def pad(self, images, division_factor=16):
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

    def image_scale_pad(self, k, img_semantic, seg_mask=False):
        # Prepare image, we scale the image with a scale factor 1/k and pad to be divisble by patch size.
        scale_factor = 1/k

        #im = img_semantic.transpose(1,2,0)
        im_dict = { "k" : img_semantic}
        self.resize(im_dict, scale_factor, seg_mask=seg_mask)
        # dont pad, we dont use patches....
        self.pad(im_dict, division_factor=32)  # we only try patch sizes 8 and 16 
        
        
        # can we simply scale the positions? yes
        #plt.imshow(im_dict["k"])
        #plt.scatter(pixel_coords[:,0] * scale_factor, pixel_coords[:,1] * scale_factor)
        
        # Put the channel as first.
        #img = torch.Tensor(im_dict["k"].transpose(2,0,1))
        
        return im_dict["k"] 
        
    def get_scene(self):
        return self.trajectories, self.trajs_df, self.coord_transform, self.img_semantic, self.img_reference, self.img_semantic_scaled, self.img_reference_scaled 
    
    
    
    

def test():
    from src.utils.constants import sdd_scenes
    eth_scene = TrajScene("sdd", sdd_scenes[20], scale_factor=8, data_root="")
    trajs, trajs_df, coord_transform, img_semantic, img_reference, img_semantic_scaled, img_reference_scaled  = eth_scene.get_scene()
    plt.imshow(img_reference_scaled)
    tr = coord_transform.get_pixel_positions(trajs[list(trajs.keys())[np.random.choice(len(trajs))]][["pos_x", "pos_y"]].values) // 8
    plt.plot(tr[:,0], tr[:,1], linewidth=3, color="red")