from Benchmarks_MADDPG import CCMADDPG
import matplotlib.pyplot as plt
from mec_env import MecEnv
import sys

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 1 
NUMBER_OF_EVAL_EPISODES = 50

DONE_PENALTY = None

ENV_SEED = 37
NUMBERofAGENTS = 50
def create_ddpg(InfdexofResult, env, env_eval, EPISODES_BEFORE_TRAIN, MAX_EPISODES, Benchmarks_mode):
    ccmaddpg = CCMADDPG(InfdexofResult=InfdexofResult, env=env, env_eval=env_eval, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound, episodes_before_train = EPISODES_BEFORE_TRAIN, epsilon_decay= MAX_EPISODES, Benchmarks_mode = Benchmarks_mode) 
    
    ccmaddpg.interact(MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES)
    return ccmaddpg
    
def plot_ddpg(ddpg, parameter, Benchmarks_modes, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_rewards, label="{}".format(Benchmarks_modes[i])) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")
                 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    if len(ddpg) == 1:
        plt.savefig('./Figures/Benchmarks_'+ Benchmarks_modes[0] +'_run%s.png'%parameter)
    else: 
        plt.savefig('./Figures/Benchmarks_run%s.png'%parameter)

def run(InfdexofResult):

    Benchmarks_modes = ["deadline_divide2_size_first_MADDPG"] #["deadline_divide2_size_first_MADDPG", "offloadtimefirst_MADDPG", "MADDPG", "mindeadlinefirst_MADDPG",  "maxdeadlinefirst_MADDPG",  "minsizefirst_MADDPG", "maxsizefirst_MADDPG"]
    env = MecEnv(n_agents=NUMBERofAGENTS, env_seed = ENV_SEED)
    eval_env = MecEnv(n_agents=NUMBERofAGENTS, env_seed = ENV_SEED) #ENV_SEED will be reset at set()
    #duplicate env so that it does not cause a poroblem in the seed for the different benchmarks
    env_list = [env]*len(Benchmarks_modes)
    eval_env_list = [eval_env]*len(Benchmarks_modes)
    ddpglist = [create_ddpg(InfdexofResult, env_list[i], eval_env_list[i], EPISODES_BEFORE_TRAIN, MAX_EPISODES, Benchmarks_modes[i]) for i in range(len(Benchmarks_modes))]

    #plot_ddpg(ddpglist, "_%s"%InfdexofResult, Benchmarks_modes)

if __name__ == "__main__":
    InfdexofResult = sys.argv[1] # set run runnumber for indexing results, 
    
    run(InfdexofResult) 
