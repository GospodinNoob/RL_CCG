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
    
    def getAction(self, observation, validActions, validEnvActions):
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
    
    def getAction(self, observation, validActions, validEnvActions):
        if len(validEnvActions) > 1:
            validActionsList = []
            for i in range(len(validActions)):
                 if (validActions[i] > 0):
                        validActionsList.append(i)
            return random.choice(validActionsList[1:])
        return 0
    
class A2Cagent(Agent):
    
    replay_capacity = 100
        
    def __init__(self, actorNetwork, valueNetwork, turn, n_actions=71, VEC_SIZE=100):
        self.n_actions = n_actions
        self.turn = turn
        self.actor_network = actorNetwork
        self.actor_network_optim = torch.optim.Adam(self.actor_network.parameters(), lr = 0.01)
        self.value_network = valueNetwork
        self.value_network_optim = torch.optim.Adam(self.value_network.parameters(), lr=0.01)
        
        self.replay = dict()
        
    def getAction(self, observation, validActions, validEnvActions):
        observation = utils.createStateObservation(observation)
        log_softmax_action = self.actor_network.get_qvalues_from_state([observation])
        softmax_action = torch.exp(log_softmax_action)
        qvalues = softmax_action.data.cpu().numpy()
        action = self.actor_network.sample_actions(qvalues, np.array([validActions]))[0]
        return action

    def record(self, replay_id, obs, action, n_obs, reward, done):
        if replay_id not in self.replay:
            
            if(len(self.replay.keys()) >= self.replay_capacity):
                self.replay.pop(random.choice(list(self.replay.keys())))
            
            self.replay[replay_id] = dict()
            self.replay[replay_id]["observations"] = []
            self.replay[replay_id]["actions"] = []
            self.replay[replay_id]["n_observations"] = []
            self.replay[replay_id]["rewards"] = []
            
        cur_replay = self.replay[replay_id]
        cur_replay["observations"].append(obs)
        cur_replay["actions"].append(action)
        cur_replay["n_observations"].append(n_obs)
        cur_replay["rewards"].append(reward)
            
        cur_replay["done"] = done
        cur_replay["final_reward"] = 0
        if not done:
            obs_main = Variable(torch.Tensor([utils.createStateObservation(n_obs)["main"]]))
            cur_replay["final_reward"] = self.value_network(obs_main).cpu().data.numpy()
    
    def getRecord(self, records_num=10):
        records = list(self.replay.items())
        replays = random.sample(self.replay.items(), min(len(records), records_num))
        obs = []
        actions = []
        rewards = []
        for i in replays:
            obs.extend(i[1]["observations"])
            actions.extend(i[1]["actions"])
            rewards.extend(i[1]["rewards"])
        
        return obs, actions, rewards
    
    def endRecord(self, replay_id):
        if(replay_id not in list(self.replay.keys())):
            return
        cur_replay = self.replay[replay_id]
        cur_replay["rewards"] = self.discount_reward(cur_replay["rewards"], 0.99, cur_replay["final_reward"])
    
    def discount_reward(self, r, gamma, final_r):
        discounted_r = np.zeros_like(r)
        for t in reversed(range(0, len(r))):
            running_add = final_r * gamma + r[t]
            discounted_r[t] = final_r
        return discounted_r
        
    def train(self):
        states, actions, rewards = self.getRecord(5)
        actions_var = Variable(torch.Tensor(actions).view(-1, self.n_actions))
        self.actor_network_optim.zero_grad()
        discount = 0.99
        log_softmax_actions = self.actor_network.get_qvalues_from_state(states)
        
        main = []
        for st in states:
            main.append(np.array(st["main"], dtype=np.float32)[None, None, :])

        states_var = Variable(torch.Tensor(np.array(main)))
        vs = self.value_network(states_var).detach()

        # calculate qs
        qs = Variable(torch.Tensor(rewards))

        advantages = qs - vs
        #print(log_softmax_actions.shape, actions_var.shape, advantages.shape)
        actor_network_loss = -torch.mean(torch.sum(log_softmax_actions * actions_var, 1) * advantages)
        #batch_loss_actor.append(actor_network_loss.detach().numpy())
        actor_network_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.actor_network.parameters(), 0.5)
        self.actor_network_optim.step()

        # train value network
        self.value_network_optim.zero_grad()
        target_values = qs
        values = self.value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        #batch_loss_value.append(value_network_loss.detach().numpy())
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(self.value_network.parameters(), 0.5)
        self.value_network_optim.step()
        
        return actor_network_loss.detach().numpy(), value_network_loss.detach().numpy()
    