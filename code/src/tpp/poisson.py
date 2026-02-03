### Poisson Process - Baseline
import numpy as np
import pandas as pd

def fit_poisson_processes(emission_sequences):
    poisson_fits = {}
    
    for k in emission_sequences.keys():
        arrival_df = emission_sequences[k]
        arrival_times = arrival_df[arrival_df.counts > 0].frame_id.values
        lambda_rate = fit_poisson(arrival_times)
        poisson_fits[k] = lambda_rate
        
    return poisson_fits
    
def generate_sequences(poisson_processes, spawns, crowd_ems, n_rounds=1, len_gen=10000):
        
    spawn_simulation = {}
    #n_rounds = 10
    #len_gen = 1000
    for k in spawns.keys():
        windows_test = sim_poisson(poisson_processes[k], len_gen*n_rounds).astype(int)
        spawn_simulation[k] = windows_test
    
    
    # Now we have a sequence of frames for each spawn.
    # What we have to do now, is count the number of occurences at each frame (because we are discrete, not continuous.)
    # Tiny hack to get multiple agents at the same time.

    # get frame emission sequence df for each cluster
    n_max_frames = n_rounds * len_gen
    simulated_emission_sequences = {}
    for k, frames in spawn_simulation.items():
        df = pd.DataFrame(np.column_stack( [frames, np.ones(len(frames)).astype(int)]), columns=["frame_id", "counts"])
        # count the frames
        df_group = df.groupby("frame_id")["counts"].count().reset_index()
        # make new df which fills the frames to a full sequence
        # we take the max frames of the dataset
        
        #df_temp = pd.DataFrame(np.arange(max(df.frame_id.values)+1), columns=["frame_id"])
        df_temp = pd.DataFrame(np.arange(n_max_frames+1), columns=["frame_id"])
        # merge and fill with 0s where no emission happened
        result_df = pd.merge(df_temp, df_group, on='frame_id', how='left')
        result_df['counts'] = result_df['counts'].fillna(0).astype(int)
        
        simulated_emission_sequences[k] = result_df
                
    # Now, we "simulate", i.e., we spawn agents.
    # --- This could be done on the fly while simulating, but we just sample some here.
    n_frames = n_rounds * len_gen
    frame_wise_spawns = []
    for frame in range(n_frames):
        
        frame_spawns = []
        for k, df in simulated_emission_sequences.items():
            #if not k in [1, 2, 6, 9]:
            #    break
            if df.counts.values[frame] > 0:
                X = spawns[k]

                candidates = X[:,0]
                n_draws = df.counts.values[frame]
                prob_distribution = X[:,2]
                
                draws = np.random.choice(candidates, n_draws, p=prob_distribution)
                pairs = np.column_stack([np.ones(len(draws)) * k, draws]).astype(int)
                
                
                #draws = np.column_stack( [crowd_ems.sample_from_pair(pairs), np.ones(len(draws)) * k])
                draws = np.column_stack( [crowd_ems.sample_from_pair(pairs), pairs])
                
                frame_spawns.append(draws)
                
        frame_wise_spawns.append(frame_spawns)

    return simulated_emission_sequences, frame_wise_spawns

def fit_poisson(arrival_times):

    # Example: Your real data
    real_arrival_times = arrival_times  # Example data in units of time

    # Calculate lambda (average rate)

    total_time = real_arrival_times[-1] - real_arrival_times[0]
    
    num_arrivals = len(real_arrival_times)
    lambda_rate = num_arrivals / total_time
    
    #print(num_arrivals, total_time)
    
    return lambda_rate

def sim_poisson(lambda_rate, total_simulation_time=10000):
    # Specify total simulation time

    # Generate inter-arrival times and arrival times
    simulated_arrival_times = []
    current_time = 0
    #print(lambda_rate)
    while current_time < total_simulation_time:
        inter_arrival_time = np.random.exponential(1/lambda_rate)
        current_time += inter_arrival_time
        if current_time < total_simulation_time:
            simulated_arrival_times.append(current_time)
    
    return np.array(simulated_arrival_times)
    