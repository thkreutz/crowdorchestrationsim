import os
from tqdm import tqdm
import torch
import distinctipy
#colors = np.array(distinctipy.get_colors(len(np.unique((df_analysis.destination_id)))))
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import env.single_agent_env as tk_env

def read_crowd_df(path="crowd_history_10k_marked.pt"):
    trajs = torch.load(path)
    #trajs = torch.load("crowd_history_LONG.pt")

    trajs_temp = [np.vstack(t) for t in trajs.values()]
    temp = np.vstack(trajs_temp)
    df = pd.DataFrame(temp, columns=["frame_id", "agent_id", "spawn_id", "destination_id", "pos_x", "pos_y", "vel_x", "vel_y"])
    df["frame_id"] = df["frame_id"].values.astype(int)
    df["agent_id"] = df["agent_id"].values.astype(int)
    df["spawn_id"] = df["spawn_id"].values.astype(int)
    return df, trajs
env = tk_env.CustomEnv(dataset_name="gc", scene_name="gc", data_root="", scale_factor=1, n_agents=1, rendering="", action_scaler=None)

df_analysis, _ = read_crowd_df("../ppt_seqs/GC/modify/crowd_history_top2_round_robin.pt")

color_spawns = True
if color_spawns:
    uniqs = np.unique(df_analysis.spawn_id)
else:
    uniqs = np.unique(df_analysis.destination_id)
    
#colors = sns.color_palette("cubehelix", len(uniqs))
colors = np.array(distinctipy.get_colors(len(uniqs)))
color_dict = { sid : colors[i] for i, sid in enumerate(uniqs) }


## For a sequence of frames, we need all agents present in the frame.
## => Get their histories, plot
df_plot = df_analysis
typ="publication_top_rr"
save_path = "../ppt_seqs/GC/modify/video/%s" % typ

if not os.path.isdir(save_path):
    os.makedirs(save_path)
    
for frame in tqdm(range(0, 26000)):
    fig = plt.figure(figsize=(20,10))
    start_frame = frame

    length = 7
    df_vis = df_plot[ (df_plot.frame_id >= start_frame) & (df_plot.frame_id <= start_frame+length) ]

    vis_trajs = [ag[["pos_x", "pos_y", "destination_id", "spawn_id"]].values for _, ag in list(df_vis.groupby("agent_id"))]
    plt.xlim((0, 1920))
    plt.ylim((1080, 0))
    
    plt.imshow(np.swapaxes(env.img_reference, 0, 1))
    for ag in vis_trajs:
        dest_color = ag[0, 3]
        ag = ag[:,:2] * 2

        #plt.scatter(ag[-1,0], ag[-1,1]) #, color=color_dict[dest_color])
        #plt.plot(ag[:,0], ag[:,1]) #, c=color_dict[dest_color])

        plt.scatter(ag[-1,0], ag[-1,1], color=color_dict[dest_color])
        plt.plot(ag[:,0], ag[:,1], c=color_dict[dest_color], alpha=.5)
        
    plt.axis("off")
    plt.savefig(os.path.join(save_path, "frame_%s.png"% frame), bbox_inches="tight")
    plt.clf()
    plt.close(fig)