import torch
import numpy as np
from src.data.GC_dataset import GC_Dataset
from src.data.Traj_dataset import TrajDataset
import copy

def min_max_scale(x):
    x_min = x.min()
    x_max = x.max()
    new_x = (x - x_min) / (x_max - x_min)
    return new_x, x_min, x_max

def min_max_scale_given(x, x_min, x_max):
    new_x = (x - x_min) / (x_max - x_min)
    return new_x

def reverse_min_max_scale(x, x_min, x_max):
    return (x * (x_max - x_min)) + x_min

class MinMaxScaler:
    def __init__(self):
        self.min_maxscale_dict = {}
    
    def fit_transform(self, x, key):
        x_, x_min, x_max = min_max_scale(x)
        self.min_maxscale_dict[key] = (x_min, x_max)
        return x_
        
    def transform(self, x, key):
        x_min, x_max = self.min_maxscale_dict[key]
        return min_max_scale_given(x, x_min, x_max)
    
    def inverse_transform(self, x, key):
        x_min, x_max = self.min_maxscale_dict[key]
        return reverse_min_max_scale(x, x_min, x_max)


def load_data(k, scale=True):
    ### k = image downscaling factor
    gc = GC_Dataset(k=k)

    coord_transform, img, DFs = gc.get_dataset("gc")
    
    ag_idxs = list(DFs.keys())
    n_removed = 0
    n_total = 0
    n_after = 0
    lens = []
    traj_dataset = {}
    for test_dataset in ["gc"]:
        coord_transform, img, DFs = gc.get_dataset(test_dataset)
        trajs = []
        for df in DFs.values():
            xx = df[["pos_x", "pos_y"]].values
            diff = (df.goal_x.values[0] - df.pos_x.values[0])**2 + (df.goal_y.values[0] - df.pos_y.values[0])**2
            
            # just make based on velocity?
            if diff > 2 and len(df) < 100: # want that agents moves a good amount of space and no ultra long trajectories   ### len(df) < 30
                trajs.append(df)
                lens.append(len(df))
        #print(len(DFs) - len(trajs))
        n_removed += (len(DFs) - len(trajs))
        n_total += len(DFs)
        n_after += len(trajs)
        traj_dataset[test_dataset] = [coord_transform, img, trajs]
        #
        #    plt.plot(xx[:,0], xx[:,1])
        #    trajs.append(xx)
        
    dataset = TrajDataset(traj_dataset, ["gc"], k=k)
    traj_dataset = dataset.traj_dataset
    coord_transform, img, trajs = traj_dataset["gc"]
    trajs_train = trajs[0]
    trajs_train = torch.Tensor(np.vstack(trajs_train))
    trajs_test = trajs[1]



    non_scaled_pos = copy.deepcopy(trajs_train[:,:,:2])
    non_scaled_trajs_train = copy.deepcopy(trajs_train)


    min_max_scaler = MinMaxScaler()
        
    if scale:
        for i, key in zip([0,1,2,3,4,5,8,9], ["pos_x", "pos_y", "vel_x", "vel_y", "goal_x", "goal_y", "dist", "ang"]):
            trajs_train[:,:,i] = min_max_scaler.fit_transform(trajs_train[:,:,i], key)
            
            
    return coord_transform, img, DFs, gc, trajs, trajs_train, non_scaled_pos, non_scaled_trajs_train, min_max_scaler