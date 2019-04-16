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
from collections import namedtuple

class Agent():
    replay = []
    
    training_state = True
    
    def _init__(self, turn):
        self.turn = turn
    
    def getAction(self, observation, validActions, validEnvActions, evaluate = False):
        pass
    
    def record(self, replay_id, obs, action, n_obs, reward, done):
        pass
    
    def endRecord(self, replay_id):
        pass
    
    def setTraning(self, state = True):
        self.training_state = state
    
    def train(self):
        return None
    
    def setTurn(self, turn):
        self.turn = turn
    
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
    
ActionResult = namedtuple("action_result",("snapshot", "observation", "reward", "is_done"))

class Node:
    """ a tree node for MCTS """
    
    parent = None          #parent Node
    value_sum = 0.         #sum of state values from all visits (numerator)
    times_visited = 0      #counter of visits (denominator)

    
    def __init__(self,parent, action, turn):
        self.parent = parent
        self.action = action   
        self.turn = turn
        self.children = set()       #set of child nodes

        #get action outcome and save it
        res = env.get_result(parent.snapshot, action)
        self.snapshot,self.observation,self.immediate_reward,self.is_done,_ = res
        
        
    def is_leaf(self):
        return len(self.children)==0
    
    def is_root(self):
        return self.parent is None
    
    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited !=0 else 0
    
    def ucb_score(self,scale=10,max_value=1e100):
        if self.times_visited == 0:
            return max_value
        
        U = (np.log(self.parent.times_visited) / (self.times_visited))**0.5
        return self.get_mean_value() + scale*U
    
    
    
    def select_best_leaf(self):
        if self.is_leaf():
            return self
        
        children = self.children
        m = None
        best_child = None
        for i in children:
            if(m == None):
                best_child = i
                m = i.ucb_score() 
            elif (i.ucb_score() > m):
                best_child = i
                m = i.ucb_score() 
        
        return best_child.select_best_leaf()
    
    def expand(self):
        for action in range(n_actions):
            self.children.add(Node(self,action))
        
        return self.select_best_leaf()
    
    def rollout(self,t_max=10):
            
        session = copy.deepcopy(self.snapshot)
        _, _, validActionsEnv = session.processNewStateInfo()
        obs = self.observation
        is_done = self.is_done
        
        totalRew = 0
        
        for i in range(t_max):
            if is_done:
                break
            obs, _ , validActionsEnv = session.action(random.choice(validActionsList))
            rew = session.getHealthAdvantage(1 - self.turn)
            is_done = observation["end"] or obs["turn"] != self.turn
            totalRew += rew

        return totalRew
    
    def propagate(self,child_value):
        my_value = self.immediate_reward + child_value
        
        self.value_sum+=my_value
        self.times_visited+=1
        
        if not self.is_root():
            self.parent.propagate(my_value)
        
    def safe_delete(self):
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child
            
class Root(Node):
    def __init__(self,snapshot = None,observation = None):
        self.parent = self.action = None
        self.children = set()
        
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done=False
    
    @staticmethod
    def from_node(node):
        root = Root(node.snapshot,node.observation)
        copied_fields = ["value_sum","times_visited","children","is_done"]
        for field in copied_fields:
            setattr(root,field,getattr(node,field))
        return root
    
class MCTSAgent(Agent):
    
    self.globalRoot = None
    
    def __init__(self, turn, n_actions=71):
        self.turn = turn
        self.globalRoot = Root()
        self.root = self.globalRoot
        
    def getAction(self, observation, validActions, validEnvActions, evaluate = False):
        m = None
        best_child = None
        children = root.children
        
        for i in children:
            if(m == None):
                best_child = i
                m = i.ucb_score() 
            elif (i.ucb_score() > m):
                best_child = i
                m = i.ucb_score()
                
        self.root = Root.from_node(best_child)
        if(self.root.is_leaf()):
            for _ in range(n_iters):
                node = root.select_best_leaf()

                if node.is_done:
                    node.propagate(0)

                else:
                    node.expand()
                    rew = node.rollout()
                    node.propagate(rew)
        return best_child.action
    
    def train(self):
        pass
    
    
    
class A2Cagent(Agent):
    
    def __init__(self, actorNetwork, valueNetwork, turn, replay ,n_actions=71, VEC_SIZE=100, epsilon = 0.2, entropy_coef = 0.01):
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
            
    def entropy(self, p):
        return -torch.sum(torch.exp(p) * p, 1)
        
    def train(self):
        if not self.training_state:
            return
        self.epsilon *= 0.999
        self.epsilon = max(0.1, self.epsilon)
        states, actions, rewards, indices, weights = self.getRecord(100)
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
    