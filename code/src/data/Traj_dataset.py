import torch
import numpy as np

class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, traj_dataset, scenes, k, split=0.8):
        self.traj_dataset = traj_dataset
        self.scenes = scenes
        self.traj_dataset = { name : (values[0], values[1], self.traj_goals(values[2], split=split)) for name, values in traj_dataset.items() if name in scenes}
        #self.traj_dataset_test = { name : (values[0], values[1], self.traj_goals(values[2], test=True) ) for name, values in traj_dataset.items() if name in sdd_scenes_test}
        self.k = k
        #self.states = {name : self.prep(name) for name in scenes}

    def traj_goals(self, trajs, split=0.8):
        data_train = []
        data_test = []
        
        # We can also shuffle here.
        
        split_idx = int(split * len(trajs))
        
        print("Training=%s, Testing=%s" % (split_idx, len(trajs)-split_idx))
        
        #### In here we now prepare to make polar coord states.

        # Polar coords are distance and angle towards to goal.
        # State at time t: d, ang, observation.
        #                  (+ velocity !!)
        # Prediction should be this: d', ang' => d = d + d_, ang = ang + ang'
        # To get this, we can either directly predict next d and ang, or we can let the model predict the offset.

        for df in trajs:
            # Each df gets a new column d, ang.
            # Agent centric => Subtract agents position at t from its goal position.
            dist, ang = self.to_polar(df.goal_x.values - df.pos_x.values, df.goal_y.values - df.pos_y.values)
            # New column for distance and angle.
            df["dist"] = dist
            df["ang"] = ang
            df["last_frame"] = df.frame_id.values[-1]

        
        #### rolling gives me a sliding window over all of the sequences.
        
        #### If we go with 5, we get a sequence of 5 future predictions that the model has to take into account.
        #### Like that we can introduce a common sense loss maybe....
        
        for df in trajs[:split_idx]:
            ws = [w.values for w in df[["pos_x", "pos_y", "vel_x", "vel_y", "goal_x", "goal_y", "start_x", "start_y", "dist", "ang",  "frame_id", "last_frame"]].rolling(2)][1:]
            data_train.append(ws)
        
        for df in trajs[split_idx:]:
            ws = [w.values for w in df[["pos_x", "pos_y", "vel_x", "vel_y", "goal_x", "goal_y", "start_x", "start_y", "dist", "ang",  "frame_id", "last_frame"]].rolling(2)][1:]
            data_test.append(ws)
            
        #if test:
        return (data_train, data_test)
        #else:
        #    return np.vstack(data)

    def __len__(self):
        return len(self.traj_dataset)
    
    def state_img(self, img, pos, goal, coord_transform, k):

        current_pos = coord_transform.get_pixel_positions_torch(pos) // k
        goal_pos = coord_transform.get_pixel_positions_torch(goal) // k

        aug = torch.zeros( (3, img.shape[1], img.shape[2]) )

        ### Add goal and start
        aug[0][current_pos[1]][current_pos[0]] = 1
        aug[1][goal_pos[1]][goal_pos[0]] = 1


        ### Relative distance to start pixel.

        # Get all pixel positions
        X, Y = np.mgrid[0:img.shape[2]:1, 0:img.shape[1]:1]
        positions = torch.Tensor(np.vstack([X.ravel(), Y.ravel()]).T)

        # To world coordinates
        positions_world = coord_transform.get_positions_torch(positions * k)

        # Inverse distance
        #agent_pos = trajs[0][0, :2]

        #rel_dists = pos - positions_world

        #rel_dists = 1 / (torch.cdist(pos.unsqueeze(0), positions_world).reshape(img.shape[2], img.shape[1]) + 0.0001)
        rel_dists = torch.cdist(pos.unsqueeze(0), positions_world).reshape(img.shape[2], img.shape[1]) # unscaled distance in world coords.

        # Scaled by max possible distance in pixel coords on the image itself
        ### rel_dists = torch.cdist(current_pos[1].float().unsqueeze(0), positions).reshape(img.shape[2], img.shape[1]) / np.sqrt(img.shape[2]**2 + img.shape[1]**2)

        aug[2] = rel_dists.T

        return torch.vstack( (img, aug))
    
    def to_polar(self, x, y):
        return np.sqrt((x**2 + y**2)), np.arctan2(y, x)

    def prep(self, scene):
        # Returns one scene
        coord_transform, img, trajs = self.traj_dataset[scene]
        # shuffle trajs
        trajs = np.vstack(trajs)
        #np.random.shuffle(trajs)
        trajs = torch.Tensor(trajs)
        
        # x,y,xg,yg to polar coordinates
        # Polar coordinates
        #goal_x = 

        #state_imgs = None
        return coord_transform, img, trajs
    
    def get_test(self, idx):
        # Returns one scene
        coord_transform, img, trajs = self.states[self.scenes_test[idx]]
        return coord_transform, img, trajs

    def __getitem__(self, idx):
        # Returns one scene
        coord_transform, img, trajs = self.states[self.scenes_train[idx]]
        return coord_transform, img, trajs