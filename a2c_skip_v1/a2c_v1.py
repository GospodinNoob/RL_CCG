import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ActorNetwork(nn.Module):

    def __init__(self, state_shape, n_actions, epsilon = 0.5):
        super(ActorNetwork, self).__init__()
        self.epsilon = epsilon
        self.layers = nn.Sequential(
            nn.Linear(state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state_t):
        qvalues = self.layers(state_t)
        #out = F.log_softmax(self.fc3(out))
        return qvalues
    
    def sample_actions(self, qvalues, valid_actions):
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        qvalues[np.logical_not(valid_actions)] = -2**32
        valid_actions = valid_actions.astype(np.int)
        valid_actions = [va / np.sum(va) for va in valid_actions]
        random_actions = [np.random.choice(n_actions, size=batch_size, p=va)[0] for va in valid_actions]
        best_actions = qvalues.argmax(axis=-1)
        
        should_explore = np.random.choice([0, 1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
    def get_qvalues(self, states):
        states = Variable(torch.FloatTensor(np.asarray(states)))
        qvalues = self.forward(states)
        return qvalues

class ValueNetwork(nn.Module):

    def __init__(self,state_shape):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state_t):
        reward = self.layers(state_t)
        reward = reward.reshape(reward.shape[0])
        return reward