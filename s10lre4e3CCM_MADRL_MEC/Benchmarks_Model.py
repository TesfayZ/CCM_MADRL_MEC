import torch as th
from torch import nn
class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w =3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out
class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w =3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)
        
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        out = th.cat([state, action], 0)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))      
        out = self.fc3(out)
        return out


