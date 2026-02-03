import copy
import numpy as np


# Augment existing trajectories by a) translation and b) adding noise
# Need to update positions, then update goal and start in df as well. 
## TODO: Filter if they collide with anything.
def augment_trajectory(df_to_aug, range_x=[-0.1, 0.1], range_y=[-0.1, 0.1], stepsize=20, noise_x=0.05, noise_y=0.05):
    # translate pos_x, pos_y
    x_min = range_x[0]
    x_max = range_x[1]
    y_min = range_y[0]
    y_max = range_y[1]
    
    ##### Can implement sampling of different start + endpoints from the trajectory in here!

    df_augs_trans = []
    trans_x, trans_y = np.linspace(x_min, x_max, stepsize), np.linspace(y_min, y_max, stepsize)
    for tx in trans_x:
        for ty in trans_y:
            df_temp = copy.deepcopy(df_to_aug)
            df_temp["pos_x"] += tx
            df_temp["pos_y"] += ty
            df_temp["goal_x"] = df_temp["pos_x"].values[-1]
            df_temp["goal_y"] = df_temp["pos_y"].values[-1]
            df_temp["start_x"] = df_temp["pos_x"].values[0]
            df_temp["start_y"] = df_temp["pos_y"].values[0]
            df_augs_trans.append(df_temp)

    df_augs_noise = []
    for tx in trans_x:
        for ty in trans_y:
            
            # different noise variations...
            for i in range(2):
                # add random noise as well so that trajectories look a bit different
                tx += np.random.normal(0,noise_x)
                ty += np.random.normal(0,noise_y)

                df_temp = copy.deepcopy(df_to_aug)
                df_temp["pos_x"] += tx
                df_temp["pos_y"] += ty
                df_temp["goal_x"] = df_temp["pos_x"].values[-1]
                df_temp["goal_y"] = df_temp["pos_y"].values[-1]
                df_temp["start_x"] = df_temp["pos_x"].values[0]
                df_temp["start_y"] = df_temp["pos_y"].values[0]
                df_augs_noise.append(df_temp)
    return df_augs_trans + df_augs_noise


#def augmentation_test():
#    #### augmentation test
#    df_auged = augment_trajectory(df_test)
#    pxll = []
#    for dft in df_auged:
#        pxl = coord_transform.get_pixel_positions(dft[["pos_x", "pos_y"]].values) / 16
#        pxll.append(pxl)
##
 #   plt.figure(figsize=(10,10))
#    pxl = coord_transform.get_pixel_positions(trajs[0][["pos_x", "pos_y"]].values) / 16
#    plt.imshow(img[0])
#    for pp in pxll:
#        plt.plot(pp[:,0], pp[:,1], c="lightgrey")##
#
#    plt.plot(pxl[:,0], pxl[:,1], c="red")