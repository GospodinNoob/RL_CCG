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

class Agent():
    replay = []
    
    def getAction(self, observation, validActions, validEnvActions):
        pass
    
    def record(self, replay_id, obs, action, n_obs, reward, done):
        pass
    
    def train(self):
        pass
    
class ARagent(Agent):
    def getAction(self, observation, validActions, validEnvActions):
        if len(validEnvActions) > 1:
            validActionsList = []
            for i in range(len(validActions)):
                 if (validActions[i] > 0):
                        validActionsList.append(i)
            return random.choice(validActionsList[1:])
        return 0
    
class A2Cagent(Agent):
        
    def __init__(self, actorNetwork, valueNetwork, n_actions=71, VEC_SIZE=100):
        self.n_actions = n_actions
        self.actor_network = actorNetwork
        self.actor_network_optim = torch.optim.Adam(self.actor_network.parameters(), lr = 0.01)
        self.value_network = valueNetwork
        self.value_network_optim = torch.optim.Adam(self.value_network.parameters(), lr=0.01)
        
        self.replay = dict()
        
    def getAction(self, observation, validActions, validEnvActions):
        observation = utils.createStateObservation(observation)
        log_softmax_action = self.actor_network.get_qvalues_from_state(observation)
        softmax_action = torch.exp(log_softmax_action)
        qvalues = softmax_action.data.cpu().numpy()
        action = self.actor_network.sample_actions(qvalues, np.array([validActions]))[0]
        return action

    def record(self, replay_id, obs, action, n_obs, reward, done):
        if replay_id not in self.replay:
            self.replay[replay_id] = dict()
            self.replay[replay_id]["observations"] = [obs]
            self.replay[replay_id]["actions"] = [action]
            self.replay[replay_id]["n_observations"] = [n_obs]
            self.replay[replay_id]["rewards"] = [reward]
            
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
    
    def getRecord(self):
        replay_id = random.choice(self.replay.keys())
        cur_replay = self.replay[replay_id]
        return cur_replay["observations"], cur_replay["actions"], cur_replay["rewards"], cur_replay["final_reward"]
    
    def discount_reward(r, gamma, final_r):
        discounted_r = np.zeros_like(r)
        running_add = final_r
        for t in reversed(range(0, len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
        
    def train(self):
        states, actions, rewards, final_r = self.getRecord()
        return
        actions_var = Variable(torch.Tensor(actions).view(-1, n_actions))
        self.actor_network_optim.zero_grad()
        main = []
        our_units = []
        enemy_units = []
        enemy_core = []
        discount = 0.99
        our_piles = []
        our_hands = []
        for st in states:
            main.append(st[0])
            our_units.append(st[1])
            enemy_units.append(st[2])
            enemy_core.append(st[3])
            our_piles.append(st[4])
            our_hands.append(st[5])
        log_softmax_actions = actor_network(np.array(main), 
                                            np.array(our_units), 
                                            np.array(enemy_units), 
                                            np.array(enemy_core), 
                                            np.array(our_piles),
                                            np.array(our_hands)
                                           )

        states_var = Variable(torch.Tensor(np.array(main)).view(-1, MAIN_SIZE))
        vs = self.value_network(states_var).detach()

        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards, discount, final_r)))

        advantages = qs - vs
        actor_network_loss = -torch.mean(torch.sum(log_softmax_actions * actions_var, 1) * advantages)
        batch_loss_actor.append(actor_network_loss.detach().numpy())
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.5)
        self.actor_network_optim.step()

        # train value network
        self.value_network_optim.zero_grad()
        target_values = qs
        values = value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        batch_loss_value.append(value_network_loss.detach().numpy())
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
        self.value_network_optim.step()
    