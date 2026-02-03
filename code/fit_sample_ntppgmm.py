from src.tpp import tpp_sequences
import importlib
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import sys
import argparse
import torch
from src.tpp import poisson
import os

### Fixed values, can experiment with these to better fit spawns/destinations
# GC values
# scale_factor=1, dbscan_eps=.3, dbscan_min_samples=15 , top_k=80
# Forum values
#  scale_factor=1, dbscan_eps=1.1, dbscan_min_samples=3, top_k = 30
# ATC values
#  scale_factor=1, dbscan_eps=1.1, dbscan_min_samples=10, top_k=100
# ETH values
#  scale_factor=1, dbscan_eps=1.1, dbscan_min_samples=2, top_k=100
# Hotel values
#  scale_factor=1, dbscan_eps=1, dbscan_min_samples=2, top_k=100


def fit_sample(args):

    dataset_folder = args.dataset
    dataset_name = dataset_folder.lower()
    scene = args.scene

    print("Preparing dataset spawn sequences..")
    if args.dataset == "GC":
        ### GC
        crowd_ems, env = tpp_sequences.fit_crowd_emissions(dataset_name=dataset_name, scene_name=scene, 
                                                    scale_factor=1, dbscan_eps=.3, dbscan_min_samples=15, top_k=80)

    elif args.dataset == "ETH" and args.scene == "eth":
        ### ETH ETH
        crowd_ems, env = tpp_sequences.fit_crowd_emissions(dataset_name=dataset_name, scene_name=scene, 
                                                    scale_factor=1, dbscan_eps=.8, dbscan_min_samples=3, top_k=30)

    elif args.dataset == "ETH" and args.scene == "hotel":
        ### ETH HOTEL
        crowd_ems, env = tpp_sequences.fit_crowd_emissions(dataset_name=dataset_name, scene_name=scene, 
                                                    scale_factor=1, dbscan_eps=1, dbscan_min_samples=2, top_k=100)

    elif args.dataset == "Forum" and args.scene == "14Jul":
        ### Forum
        #dataset_folder = "Forum"
        #dataset_name = dataset_folder.lower()
        #scene = "14Jul"

        crowd_ems, env = tpp_sequences.fit_crowd_emissions(dataset_name=dataset_name, scene_name=scene, 
                                                    scale_factor=1, dbscan_eps=2, dbscan_min_samples=5, top_k=100)

    else:
        print("Dataset/Scene combination not supported, please add this combination to the implementation.")
        sys.exit(0)

    print("Fitting nTPP-GMM")
    #1. Fit nTPP and generate sequence
    ### Now we fit a tpp on respective spawns
    spawn_tpps, spawns = tpp_sequences.fit_tpps_on_dataset(crowd_ems, wsize=args.wsize, overlap=args.overlap, max_epochs=args.epochs, device=args.device)
    ### We can now generate several sequences
    simulated_emission_sequences, frame_wise_spawns = tpp_sequences.generate_sequences(spawn_tpps, spawns, crowd_ems, n_rounds=args.n_rounds, len_gen=args.len_gen, device=args.device)
    
    if not os.path.exists("../ppt_seqs/%s" % (dataset_folder)):
        os.makedirs("../ppt_seqs/%s" % (dataset_folder))
    
    torch.save(frame_wise_spawns, "../ppt_seqs/%s/%s_ppt_seq.pt" % (dataset_folder, scene) )

    print("Fitting Poisson-GMM")
    #2. Fit Poisson process and generate sequence
    ### Poisson Process
    poisson_processes = poisson.fit_poisson_processes(crowd_ems.emission_sequences)
    simulated_emission_sequences_poisson, frame_wise_spawns_poisson = poisson.generate_sequences(poisson_processes, spawns, crowd_ems, n_rounds=args.n_rounds, len_gen=args.len_gen)
    torch.save(frame_wise_spawns_poisson, "../ppt_seqs/%s/%s_poisson_seq.pt" % (dataset_folder,scene))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',  type=str, default='ETH', help="name of the run")
    parser.add_argument('--scene', '-s', type=str, default="eth")
    parser.add_argument('--epochs', '-e',  type=int, default=500, help="number of epochs")
    parser.add_argument('--wsize', '-w',  type=int, default=1000, help="wsize to fit")
    parser.add_argument('--overlap', '-o',  type=int, default=50, help="overlap")
    parser.add_argument('--n_rounds', '-nr',  type=int, default=1, help="number of rounds to sample")
    parser.add_argument('--len_gen', '-l',  type=int, default=1000, help="max time to generate samples")
    parser.add_argument('--device', '-de',  type=str, default="cuda", help="device for ntpp")
    args = parser.parse_args()

    fit_sample(args)