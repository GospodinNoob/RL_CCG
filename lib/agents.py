import numpy as np
import copy
import random
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Agent(nn.Module):
    replay = None
    
    def getAction(observation, validEnvActions, validActions):
        pass
    
    def record(obs, action, n_obs, reward, done):
        pass
    
    def train():
        pass
    
class ARagent(Agent):
    def getAction(observation, validEnvActions, validActions):
        if len(validEnvActions) > 1:
            return random.choice(validEnvActions[1:])
        return validEnvActions[0]
    
class A2Cagent(Agent):
    
    class ValueNetwork(nn.Module):
        def __init__(self, MAIN_SIZE):
            super(ValueNetwork, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(MAIN_SIZE, 256),
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
    