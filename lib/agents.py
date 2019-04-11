import numpy as np
import copy
import random
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ccg
import utils
from IPython.display import clear_output

class Agent():
    replay = []
    
    def _init__(self, turn):
        self.turn = turn
    
    def getAction(self, observation, validActions, validEnvActions, evaluate = False):
        pass
    
    def record(self, replay_id, obs, action, n_obs, reward, done):
        pass
    
    def endRecord(self, replay_id):
        pass
    
    def train(self):
        return None
    
class SkipAgent(Agent):
    
    def __init__(self, turn):
        self.turn = turn
    
    def getAction(self, observation, validActions, validEnvActions):
        return 0
    
class ARagent(Agent):
    
    def __init__(self, turn):
        self.turn = turn
    
    def getAction(self, observation, validActions, validEnvActions, evaluate = False):
        if len(validEnvActions) > 1:
            validActionsList = []
            for i in range(len(validActions)):
                 if (validActions[i] > 0):
                        validActionsList.append(i)
            return random.choice(validActionsList[1:]), [0]
        return 0, [0]
    
class A2Cagent(Agent):
    
    def __init__(self, actorNetwork, valueNetwork, turn, replay ,n_actions=71, VEC_SIZE=100, epsilon = 0.2, entropy_coef = 100):
        self.n_actions = n_actions
        self.turn = turn
        self.epsilon = epsilon
        self.actor_network = actorNetwork
        #self.actor_network_optim = torch.optim.Adam(self.actor_network.parameters(), lr = 0.01)
        self.actor_network_optim = torch.optim.RMSprop(self.actor_network.parameters(), lr = 0.01)
        self.value_network = valueNetwork
        self.value_network_optim = torch.optim.Adam(self.value_network.parameters(), lr=0.01)
        self.replay = replay
        self.entropy_coef = entropy_coef
        
    def getAction(self, observation, validActions, validEnvActions, evaluate = False):
        observation = utils.createStateObservation(observation)
        log_probs = self.actor_network.get_qvalues_from_state([observation])
        
        qvalues = log_probs.data.cpu().numpy()
        action = self.actor_network.sample_actions(np.exp(qvalues), np.array([validActions]), evaluate = evaluate, epsilon = self.epsilon)[0]
        return action, log_probs.detach().numpy()

    def record(self, replay_id, obs, action, n_obs, reward, done):
        final_r = 0
        if not done:
            obs_main = Variable(torch.Tensor([utils.createStateObservation(n_obs)["main"]]))
            final_r = self.value_network(obs_main).cpu().data.numpy()
        self.replay.record(obs, action, n_obs, reward, final_r)
    
    def getRecord(self, records_num=10):
        return self.replay.sample(records_num)
    
    def endRecord(self, replay_id):
        self.replay.endRecord()
        
    def softUpdate(self, target, source, soft_tau = 0.5):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
            
    def entropy(self, p):
        return -torch.sum(torch.exp(p) * p, 1)
        
    def train(self):
        self.epsilon *= 0.999
        self.epsilon = max(0.1, self.epsilon)
        states, actions, rewards, indices, weights = self.getRecord(400)
        actions_var = Variable(torch.Tensor(actions).view(-1, self.n_actions))
        self.actor_network_optim.zero_grad()
        action_log_probs = self.actor_network.get_qvalues_from_state(states)
        
        main = []
        for st in states:
            main.append(np.array(st["main"], dtype=np.float32)[None, None, :])

        states_var = Variable(torch.Tensor(np.array(main)))
        vs = self.value_network(states_var).detach()

        # calculate qs
        qs = Variable(torch.Tensor(rewards))
        entropy_loss = torch.mean(self.entropy(action_log_probs))
        advantages = qs - vs
        weights = Variable(torch.FloatTensor(weights))
        actor_network_loss = torch.sum(action_log_probs * actions_var, 1) 
        actor_network_loss *= advantages * weights# * 0.0005
        prios = torch.abs(advantages) + 1e-5
        actor_network_loss = -torch.mean(actor_network_loss)# - entropy_loss * self.entropy_coef
        actor_network_loss.backward()
        #actor_network_loss.backward()
        self.replay.update_priorities(indices, prios.data.cpu().numpy())
        torch.nn.utils.clip_grad_norm(self.actor_network.parameters(), 0.5)
        self.actor_network_optim.step()

        # train value network
        self.value_network_optim.zero_grad()
        target_values = qs
        values = self.value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(self.value_network.parameters(), 0.5)
        self.value_network_optim.step()
        
        return actor_network_loss.detach().numpy(), value_network_loss.detach().numpy()
    