import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def read_data(plot_prefix, runs, minindex):
    all_data = []
    for run_number in range(runs):
        filename = f'./CSV/AtTraining/{plot_prefix}'+str(run_number+minindex)+'.csv'
        data = pd.read_csv(filename, header=None).values.flatten()
        all_data.append(data)

        print(f"{plot_prefix} run {run_number} data length: {len(data)}")

    if not all_data:
        #print(f"Error: No valid data found in any {plot_prefix}_data.")
        return None

    shortest_length = min(len(data) for data in all_data)
    all_data = [data[:shortest_length] for data in all_data]
    return all_data, shortest_length

def plot95_ddpg(output_dir, Benchmark_modes, runs, minindex, variable="reward", confidence_interval=0.95):
    plt.figure()
    for plot_prefix in Benchmark_modes:
        all_data, shortest_length = read_data(plot_prefix, runs, minindex)
        if all_data is None:
            return

        mean_rewards = np.nanmean(all_data, axis=0)
        std = np.nanstd(all_data, axis=0)
        n = len(all_data)
        ste = std / np.sqrt(n)
        conf_int = ste * scipy.stats.t.ppf((1 + confidence_interval) / 2, n - 1)

        episodes = range(1, shortest_length + 1)

        plt.plot(episodes, mean_rewards, label=f"{plot_prefix}")
        plt.fill_between(episodes, mean_rewards - conf_int, mean_rewards + conf_int, alpha=0.3)
    plt.title("Performacne with the training environment")
    plt.xlabel("Training episodes")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f'{output_dir}/AtTraining_s10lre2e4b3.2.png')
    plt.close()

output_dir = "./Figures"
runs = 10
minindex = 0
Benchmark_modes = ["CCM_MADRL","deadline_divide2_size_first_MADDPG", "offloadtimefirst_MADDPG", "MADDPG"]# "mindeadlinefirst_MADDPG",  "maxdeadlinefirst_MADDPG",  "minsizefirst_MADDPG", "maxsizefirst_MADDPG",  "offloadtimefirst_MADDPG", "vanilla_MADDPG"]
plot95_ddpg(output_dir, Benchmark_modes, runs, minindex)
