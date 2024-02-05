from CCM_MADRL import CCM_MADDPG
import matplotlib.pyplot as plt
from mec_env import MecEnv
import sys

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 1 
NUMBER_OF_EVAL_EPISODES = 50

DONE_PENALTY = None

ENV_SEED = 37
NUMBERofAGENTS = 50
def create_ddpg(InfdexofResult, env, env_eval, EPISODES_BEFORE_TRAIN, MAX_EPISODES):
    ccmaddpg = CCM_MADDPG(InfdexofResult=InfdexofResult, env=env, env_eval=env_eval, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, 
                  action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound, episodes_before_train = EPISODES_BEFORE_TRAIN, epsilon_decay= MAX_EPISODES) 
                  
    ccmaddpg.interact(MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES)
    return ccmaddpg
    
def plot_ddpg(ddpg, parameter, variable="reward"):
    plt.figure()
    if (variable == "reward"):
        for i in range(len(ddpg)):
            plt.plot(ddpg[i].episodes, ddpg[i].mean_rewards) 
            plt.xlabel("Episode")
            plt.ylabel("Reward")
                   
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(["CCMMADDPG"])
    plt.savefig("./Figures/CCMaDDPG_run%s.png"%parameter)

def run(InfdexofResult):    
    env = MecEnv(n_agents=NUMBERofAGENTS, env_seed = ENV_SEED)
    eval_env = MecEnv(n_agents=NUMBERofAGENTS, env_seed = ENV_SEED) #ENV_SEED will be reset at set()
    ddpg = [create_ddpg(InfdexofResult, env, eval_env, EPISODES_BEFORE_TRAIN, MAX_EPISODES)]
    plot_ddpg(ddpg, "_%s"%InfdexofResult)

if __name__ == "__main__":
    InfdexofResult = sys.argv[1] # set run runnumber for indexing results, 
    run(InfdexofResult) 
