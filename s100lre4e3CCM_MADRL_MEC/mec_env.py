import numpy as np
from copy import deepcopy
LAMBDA_E = 0.5
LAMBDA_T = 0.5
MIN_SIZE = 1 #MB*1024*8
MAX_SIZE = 50 #MB*1024*8 #bits
MAX_CYCLE = 737.5 # cycles as in reference 
MIN_CYCLE = 300 # cycles (customized)
MIN_DDL = 0.1 # seconds
MAX_DDL = 1 # seconds
MIN_RES = 0.4 #GHz*10**9 #cycles per second
MAX_RES = 1.5 #GHz*10**9 #cycles per second
MAX_POWER = 24 # 10**(24/10) # 24 dBm converting 24 dB to watt(j/s)
MIN_POWER = 1 # 10**(1/10) # converting 1 dBm to watt(j/s)
CAPABILITY_E = 4#16.5 #GHz*10**9 #cycles per second
K_ENERGY_LOCAL = 5 * 1e-27 # no conversion
# maximum battery capacity and harvesting rate of devices
MAX_ENE =  0.5000001#3.2 # MJ*10**6 # in joules
MIN_ENE = 0.5 # MJ*10**6 # in joules
HARVEST_RATE = 0.001 # in joules

# to be checked for unit
MAX_GAIN = 14 # dB no units actually but conver to linear if it is dB but not dBm
MIN_GAIN = 5 # no units actually but conver to linear if it is dB
#NOISE_VARIANCE = 100 dBm  # if dB convert it to Watt, say that the gian is already divided by \ro, in the where part of the shannon
W_BANDWIDTH = 40 #MHZ

#server constraints
K_CHANNEL = 10 # number of channels
S_E = 400 #MB*1024*8 # server storage in MB, converted to bits
N_UNITS = 8 # number of processing units at server

ENV_MODE = "H2" # ["H2", "TOBM"]

MAX_STEPS = 100
class MecEnv(object):
    def __init__(self, n_agents, env_seed = None):     
        if env_seed is not None: 
            np.random.seed(env_seed)
        self.state_size = 7
        self.action_size = 3
        self.n_agents = n_agents
        self.W_BANDWIDTH = W_BANDWIDTH
        # state
        self.S_power = np.zeros(self.n_agents)
        self.Initial_energy = np.zeros(self.n_agents)
        self.S_energy = np.zeros(self.n_agents)
        self.S_gain = np.zeros(self.n_agents)
        self.S_size = np.zeros(self.n_agents)
        self.S_cycle = np.zeros(self.n_agents)  
        self.S_ddl = np.zeros(self.n_agents)
        self.S_res = np.zeros(self.n_agents)
        self.action_lower_bound = [0,  0.01, 0.01] #[0,  MIN_RES, MIN_POWER]
        self.action_higher_bound = [1, 1, 1] #[1, MAX_RES, MAX_POWER]        
        for n in range(self.n_agents):
            self.S_power[n] = np.random.uniform(MIN_POWER, MAX_POWER)
            self.Initial_energy[n] = np.random.uniform(MIN_ENE, MAX_ENE)
            self.S_gain[n] = np.random.uniform(MIN_GAIN, MAX_GAIN)
            self.S_res[n] = np.random.uniform(MIN_RES, MAX_RES)
    def reset_mec(self, eval_env_seed = None):# reset is used for the evaluating environment
        if eval_env_seed is not None:
            np.random.seed(eval_env_seed)
        self.step = 0
        for n in range(self.n_agents):
            self.S_size[n] =  np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL - MAX_DDL/10)
            self.S_energy[n] = deepcopy(self.Initial_energy[n])
        self.S_energy = np.clip(self.S_energy, MIN_ENE, MAX_ENE)
        State_ = []
        State_ = [[self.S_power[n], self.S_gain[n], self.S_energy[n], self.S_size[n], self.S_cycle[n], \
            self.S_ddl[n], self.S_res[n]] for n in range(self.n_agents)]
        State_ = np.array(State_)
        return State_
    def step_mec(self, action):
        # action
        A_decision = np.zeros(self.n_agents)
        A_res = np.zeros(self.n_agents)
        A_power = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            A_decision[n] = action[n][0]  # offload
            A_res[n] = self.S_res[n]*10**9*action[n][1] # resource
            A_power[n] = 10**((self.S_power[n]-30)/10)*action[n][2] # power    
        x_n= A_decision             
        DataRate  = self.W_BANDWIDTH*10**6*np.log(1 + A_power * 10**(self.S_gain/10)) / np.log(2) #
        DataRate  = DataRate/K_CHANNEL  #because bandwidth is divided equallly to the channels
        Time_proc =  self.S_size*8*1024*self.S_cycle / (CAPABILITY_E*10**9)
        Time_local = self.S_size*8*1024*self.S_cycle / (A_res)
        Time_max_local = self.S_size*8*1024*self.S_cycle / (MIN_RES*10**9)
        Time_off = self.S_size*8*1024/ DataRate 
        for i in range(x_n.size):# for the vanilla MADDPG, when it is using punishment instead of heuristic decision
            if x_n[i] == 2: # for the vanilla MADDPG benchmark 
                Time_off[i] = MAX_DDL
                x_n[i] = 1
        Time_finish = np.zeros(self.n_agents)
        #print("ENV_MODE:", ENV_MODE)
        if ENV_MODE == "H2": # The hybrid actor critic mode as in the paper in reference number 
            SortedOff = np.argsort(Time_off)
            MECtime = np.zeros(N_UNITS) #0
            counting = 0
            for i in range(self.n_agents):
                if x_n[SortedOff[i]]==1 and counting < N_UNITS: # for the first offloaded tasks, the server units are free for them
                    Time_finish[SortedOff[i]] = Time_off[SortedOff[i]] + Time_proc[SortedOff[i]]
                    MECtime[np.argmin(MECtime)] = Time_off[SortedOff[i]] + Time_proc[SortedOff[i]]
                    counting += 1
                elif x_n[SortedOff[i]]==1: # if offloaded only
                    for j in range(i) :# they are already sorted but some of them are x_n = 0, no problem, i was skipping
                        if x_n[SortedOff[j]]==1:
                            MECtime[np.argmin(MECtime)] += Time_proc[SortedOff[j]]
                    Time_finish[SortedOff[i]] = max(Time_off[SortedOff[i]], np.min(MECtime)) + Time_proc[SortedOff[i]]
                    MECtime[np.argmin(MECtime)] = max(Time_off[SortedOff[i]], np.min(MECtime)) + Time_proc[SortedOff[i]]  # update it to the finishing time of the task itself, not the finish time solely computed by finishing time of others. Bcz the offload time of task can be large
            Time_n = (1 - x_n) * Time_local + x_n *Time_finish ###########################
        elif ENV_MODE == "TOBM": # the concurrent processing mode 
            Time_n = (1 - x_n) * Time_local + x_n * (Time_off + Time_proc) ########################### only one machine for one task
        else:
            print(ENV_MODE, " is unknown")
            exit()
        #print("Time_finish ", Time_finish)   
        Time_n = [min(t, MAX_DDL)/MAX_DDL for t in Time_n] # stops process if exceeds max allowed time
        T_mean = np.mean(Time_n)
        #print("max min Time_n = ", max(Time_n), min(Time_n)) 
        Energy_local = K_ENERGY_LOCAL * self.S_size*8*1024*self.S_cycle* (A_res) 
        Energy_max_local = K_ENERGY_LOCAL * self.S_size*8*1024*self.S_cycle* (self.S_res*10**9) 
        Energy_off = A_power* Time_off 
        #print(Energy_local)
        Energy_n = (1 - x_n) * Energy_local + x_n * Energy_off #
        #print("max min Energy_n = ", max(Energy_n), min(Energy_n))
        #print("self.S_energy = ", self.S_enery)
        #print("harvest = ", np.random.normal(HARVEST_RATE,0, size=self.n_agents))
        self.S_energy = np.clip(self.S_energy - Energy_n*1e-6 + np.random.normal(HARVEST_RATE, 0, size=self.n_agents)*1e-6, 0, MAX_ENE)
        #print("S_enery = ", S_enery)
        #now, for enery <=0 set max time to max ddl for punishment
        for i in range(x_n.size):
            if self.S_energy[i]<= 0:
                Time_n[i] = MAX_DDL/MAX_DDL
        Time_penalty = np.maximum((Time_n - self.S_ddl/MAX_DDL), 0) 
        Energy_penalty = np.maximum((MIN_ENE - self.S_energy), 0)*10**6 
        time_penalty_nonzero_count = np.count_nonzero(Time_penalty)/self.n_agents
        energy_penalty_nonzero_count = np.count_nonzero(Energy_penalty)/self.n_agents
        #Reward = LAMBDA_E * ((Energy_max_local - Energy_n) / Energy_max_local - Energy_penalty) + LAMBDA_T* ((Time_max_local - Time_n) / Time_max_local - Time_penalty)  
        #Reward = LAMBDA_E * ((Energy_max_local - Energy_n) / Energy_max_local) + LAMBDA_T* ((Time_max_local - Time_n) / Time_max_local)   
        Reward = -1*(LAMBDA_E *np.array(Energy_n) + LAMBDA_T*np.array(Time_n)) -1*(LAMBDA_E *np.array(Energy_penalty) + LAMBDA_T*np.array(Time_penalty))
        Reward  = np.ones_like(Reward)*np.sum(Reward)    
        #self.S_energy = S_energy
        for n in range(self.n_agents):# new tasks
            self.S_size[n] =  np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL - MAX_DDL/10)
        #assert np.all(self.S_ddl < MAX_DDL), "Not all elements are less than MAX_DDL"
        State_ = []
        State_ = [[self.S_power[n], self.S_gain[n], self.S_energy[n], self.S_size[n], self.S_cycle[n], \
            self.S_ddl[n], self.S_res[n]] for n in range(self.n_agents)]
        State_ = np.array(State_)
        #print("State_",State_)
        self.step += 1
        done = False
        if self.step >= MAX_STEPS:
            self.step = 0
            done = True
        return State_, Reward, done, energy_penalty_nonzero_count, time_penalty_nonzero_count