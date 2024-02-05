import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def read_data(plot_prefix, runs, minindex):
    all_result_data = []
    all_constraint_data = []
    for run_number in range(runs):
        filename = f'./CSV/results/{plot_prefix}'+str(run_number+minindex)+'.csv'
        result_data = pd.read_csv(filename, header=None).values.flatten()
        all_result_data.append(result_data)
        filename = f'./CSV/Server_constraints/{plot_prefix}'+str(run_number+minindex)+'.csv'
        constraint_data = pd.read_csv(filename, header=None).values.flatten()
        all_constraint_data.append(constraint_data)
        print(f"{plot_prefix} run {run_number} data length: {len(result_data)}")

    if not all_result_data:
        #print(f"Error: No valid data found in any {plot_prefix}_data.")
        return None

    shortest_length = min(len(data) for data in all_result_data)
    all_result_data = [data[:shortest_length] for data in all_result_data]
    all_constraint_data = [data[:shortest_length] for data in all_constraint_data]
    return all_result_data, all_constraint_data, shortest_length

def plot95_ddpg(output_dir, parameter, parameterlist, Benchmark_modes, runs, minindex, variable="reward", confidence_interval=0.95):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # Create two subplots (2 rows, 1 column)
    
    for plot_prefix in Benchmark_modes:
        all_result_data, all_constraint_data, shortest_length = read_data(plot_prefix, runs, minindex)
        if all_result_data is None:
            return

        mean_rewards = np.nanmean(all_result_data, axis=0)
        std = np.nanstd(all_result_data, axis=0)
        n = len(all_result_data)
        ste = std / np.sqrt(n)
        conf_int = ste * scipy.stats.t.ppf((1 + confidence_interval) / 2, n - 1)

        episodes = range(1, shortest_length + 1)

        # Plot reward on the first subplot
        axes[0].plot(episodes, mean_rewards, label=f"{plot_prefix}")
        axes[0].fill_between(episodes, mean_rewards - conf_int, mean_rewards + conf_int, alpha=0.3)
        
        # Now, plot constraints on the second subplot
        # Assuming you have a function to calculate mean_constraints and conf_int_constraints
        mean_constraints = np.nanmean(all_constraint_data, axis=0)
        std_constraints = np.nanstd(all_constraint_data, axis=0)
        n_constraints = len(all_constraint_data)
        ste_constraints = std_constraints / np.sqrt(n_constraints)
        conf_int_constraints = ste_constraints * scipy.stats.t.ppf((1 + confidence_interval) / 2, n_constraints - 1)
        
        axes[1].plot(episodes, mean_constraints, label=f"{plot_prefix}")
        axes[1].fill_between(episodes, mean_constraints - conf_int_constraints, mean_constraints + conf_int_constraints, alpha=0.3)

    axes[0].set_xlabel("Training episodes")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    axes[1].set_xlabel("Training episodes")
    axes[1].set_ylabel("Server constraint")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.savefig(f'{output_dir}/plot_with_ci'+str(minindex)+'.png')
    plt.close()

output_dir = "./Figures"
parameter = "learning_rate"
parameterlist = [0.001, 0.01, 0.1]
runs = 10
minindex = 0
Benchmark_modes = ["CCM_MADRL","deadline_divide2_size_first_MADDPG", "offloadtimefirst_MADDPG", "MADDPG"]# "mindeadlinefirst_MADDPG",  "maxdeadlinefirst_MADDPG",  "minsizefirst_MADDPG", "maxsizefirst_MADDPG",  "offloadtimefirst_MADDPG", "vanilla_MADDPG"]
plot95_ddpg(output_dir, parameter, parameterlist, Benchmark_modes, runs, minindex)
