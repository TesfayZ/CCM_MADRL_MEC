import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def read_data(plot_prefix, runs, minindex, data_dir):
    all_result_data = []
    all_constraint_data = []
    for run_number in range(runs):
        filename = f'{data_dir}/{plot_prefix}{run_number+minindex}.csv'
        result_data = pd.read_csv(filename, header=None).values.flatten()
        all_result_data.append(result_data)
        print(f"{plot_prefix} run {run_number} data length: {len(result_data)}")

    if not all_result_data:
        return None

    shortest_length = min(len(data) for data in all_result_data)
    all_result_data = [data[:shortest_length] for data in all_result_data]
    return all_result_data, shortest_length

def plot95_ddpg(output_dir, Benchmark_modes, runs, minindex, variable="reward", confidence_interval=0.95):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # Create four subplots (4 rows, 1 column)
    subplot_labels = np.array(["A)","B)","C)"])
    for i,plot_prefix in enumerate(Benchmark_modes):
        #reward directory
        reward_data_dir = './CSV/results'
        #constraints directory
        time_constraints_data_dir = './CSV/Time_constraints'
        energy_constraints_data_dir = './CSV/Energy_constraints'
                
        all_reward, shortest_length = read_data(plot_prefix, runs, minindex, reward_data_dir)
        all_time_constraints, shortest_length_time_constraints = read_data(plot_prefix, runs, minindex, time_constraints_data_dir)
        all_energy_constraints, shortest_length_energy_constraints = read_data(plot_prefix, runs, minindex, energy_constraints_data_dir)

        if all_reward is None:
            return

        mean_rewards = np.nanmean(all_reward, axis=0)
        std = np.nanstd(all_reward, axis=0)
        n = len(all_reward)
        ste = std / np.sqrt(n)
        conf_int = ste * scipy.stats.t.ppf((1 + confidence_interval) / 2, n - 1)

        episodes = range(1, shortest_length + 1)
        # Plot reward on the first subplot
        #print(subplot_labels[0,0])
        axes[0].set_title(subplot_labels[0], loc='left', pad=5, fontsize=14)        
        axes[0].plot(episodes, mean_rewards, label=f"{plot_prefix}")
        axes[0].fill_between(episodes, mean_rewards - conf_int, mean_rewards + conf_int, alpha=0.3)

        
        # Plot time constraints on the second subplot
        mean_time_constraints = np.nanmean(all_time_constraints, axis=0)
        std_time_constraints = np.nanstd(all_time_constraints, axis=0)
        n_time_constraints = len(all_time_constraints)
        ste_time_constraints = std_time_constraints / np.sqrt(n_time_constraints)
        conf_int_time_constraints = ste_time_constraints * scipy.stats.t.ppf((1 + confidence_interval) / 2, n_time_constraints - 1)
        axes[1].set_title(subplot_labels[1], loc='left', pad=5, fontsize=14)
        axes[1].plot(episodes, mean_time_constraints, label=f"{plot_prefix}")
        axes[1].fill_between(episodes, mean_time_constraints - conf_int_time_constraints, mean_time_constraints + conf_int_time_constraints, alpha=0.3)
        
        # Plot energy constraints on the third subplot
        mean_energy_constraints = np.nanmean(all_energy_constraints, axis=0)
        std_energy_constraints = np.nanstd(all_energy_constraints, axis=0)
        n_energy_constraints = len(all_energy_constraints)
        ste_energy_constraints = std_energy_constraints / np.sqrt(n_energy_constraints)
        conf_int_energy_constraints = ste_energy_constraints * scipy.stats.t.ppf((1 + confidence_interval) / 2, n_energy_constraints - 1)
        axes[2].set_title(subplot_labels[2], loc='left', pad=5, fontsize=14)
        axes[2].plot(episodes, mean_energy_constraints, label=f"{plot_prefix}")
        axes[2].fill_between(episodes, mean_energy_constraints - conf_int_energy_constraints, mean_energy_constraints + conf_int_energy_constraints, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Training episodes")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    axes[0].set_title("Performance using evaluation environment")
    axes[1].set_title("Number of Tasks that exceeded their deadline")
    axes[2].set_title("Number of UDs that exceeded minimum battery level")

    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("% of tasks")
    axes[2].set_ylabel("% of UDs")
    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.savefig(f'{output_dir}/P3AtEval_s10.png')
    plt.close()

# Rest of the script remains unchanged

# Example usage:
output_dir = "./Figures"
runs = 10
print("Plotting for {} runs".format(runs))
minindex = 0
Benchmark_modes = ["CCM_MADRL", "deadline_divide2_size_first_MADDPG", "offloadtimefirst_MADDPG", "MADDPG"]
plot95_ddpg(output_dir, Benchmark_modes, runs, minindex)
