#https://github.com/rlcode/per
import random
import numpy as np
from SumTree import SumTree

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
    def addorupdate(self, error, sample):
        dsearch = []
        update = 0
        self.add(error, sample, dsearch, update)# add as new entry     
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
    
    def add(self, error, sample, dindex, update):
        p = self._get_priority(error)
        self.tree.add(p, sample, dindex, update)

    def sample(self, n):
        batch = []
        idxs = []
        is_weight = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        #print("priorities =",priorities)
        #sampling_probabilities = priorities/ self.tree.total()
        priorities_arr = np.array(priorities)
        sampling_probabilities = (priorities_arr + 1e-9) / (self.tree.total() + 1e-9)
        #print("sampling_probabilities =",sampling_probabilities)
        #zero=0
        #if self.tree.total()!=0 and min(priorities)!=0:
        #zero=1
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        #print("is_weight = ",is_weight)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
