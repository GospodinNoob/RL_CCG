import random
import numpy as np
from operator import itemgetter 

class Replay:
    
    cur_replay = None
    
    def __init__(self, capacity, gamma=0.99):
        self.capacity = capacity
        self.gamma = gamma
        pass
    
    def record(self, obs, action, n_obs, reward, final_r):
        if self.cur_replay == None:
            self.cur_replay = dict()
            self.cur_replay["observations"] = []
            self.cur_replay["actions"] = []
            self.cur_replay["rewards"] = []
            
        self.cur_replay["observations"].append(obs)
        self.cur_replay["actions"].append(action)
        self.cur_replay["rewards"].append(reward)
        self.cur_replay["final_reward"] = final_r
    
    def endRecord(self):
        pass
    
    def sample(self, records_num):
        pass
    
    def discount_reward(self, r, gamma, final_r):
        discounted_r = np.zeros_like(r)
        for t in reversed(range(0, len(r))):
            running_add = final_r * gamma + r[t]
            discounted_r[t] = final_r
        return discounted_r
    
class FlatReplay(Replay):
    
    records = [[], [], []]
    capacity = 1000
    gamma = 0.99
    
    def __init__(self, capacity = 2000, gamma=0.99):
        self.capacity = capacity
        self.gamma = gamma
    
    def endRecord(self):
        if(self.cur_replay == None):
            return
        
        self.cur_replay["rewards"] = self.discount_reward(self.cur_replay["rewards"], self.gamma, self.cur_replay["final_reward"])
        self.records[0].extend(self.cur_replay["observations"])
        self.records[1].extend(self.cur_replay["actions"])
        self.records[2].extend(self.cur_replay["rewards"])
        if(len(self.records[0]) > self.capacity):
            self.records[0] = self.records[0][-self.capacity:]
            self.records[1] = self.records[1][-self.capacity:]
            self.records[2] = self.records[2][-self.capacity:]
        self.cur_replay = None
    
    def sample(self, records_num):
        replays_ids = random.sample(range(len(self.records)), min(len(self.records), records_num))
       
        obs = itemgetter(*replays_ids)(self.records[0])
        actions = itemgetter(*replays_ids)(self.records[1])
        rewards = itemgetter(*replays_ids)(self.records[2])
        return obs, actions, rewards

class PrioritizedReplay(Replay):
    def __init__(self, capacity=2000, prob_alpha=0.6, gamma=0.99):
        self.prob_alpha = prob_alpha
        self.gamma = gamma
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def endRecord(self):
        if self.cur_replay == None:
            return
        self.cur_replay["rewards"] = self.discount_reward(self.cur_replay["rewards"], self.gamma, self.cur_replay["final_reward"])
        
        self.push(self.cur_replay)
        
        self.cur_replay = None
    
    def push(self, trajectory):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
        else:
            self.buffer[self.pos] = trajectory
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        observations = []
        actions = []
        rewards = []
        
        flatten_weights = []
        flatten_ids = []
        
        for i , trajectory in enumerate(samples):
            l = len(trajectory["observations"])
            flatten_weights.extend([weights[i]] * l)
            flatten_ids.extend([indices[i]] * l)
            observations.extend(trajectory["observations"])
            actions.extend(trajectory["actions"])
            rewards.extend(trajectory["rewards"])
        
        return observations, actions, rewards, flatten_ids, flatten_weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

    