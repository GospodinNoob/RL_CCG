import numpy as np
import copy
import random
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ccg

class Agent():
    replay = []
    
    def getAction(self, observation, validEnvActions, validActions):
        pass
    
    def record(self, obs, action, n_obs, reward, done):
        pass
    
    def train(self):
        pass
    
class ARagent(Agent):
    def getAction(self, observation, validActions, validEnvActions):
        if len(validEnvActions) > 1:
            return random.choice(validEnvActions[1:])
        return validEnvActions[0]
    
class A2Cagent(Agent):
        
    def __init__(self, actorNetwork, valueNetwork, n_actions=71, VEC_SIZE=100):
        self.n_actions = n_actions
        self.actor_network = actorNetwork
        self.actor_network_optim = torch.optim.Adam(self.actor_network.parameters(), lr = 0.01)
        self.value_network = valueNetwork
        self.value_network_optim = torch.optim.Adam(self.value_network.parameters(), lr=0.01)
        
        self.replay = dict()
        
    def observationMinion(self, minion):
        state = list(minion)
        state[5] = int(state[5])
        return np.array(state)
    
    def observationTable(self, table, turn):
        tables = copy.deepcopy(table)
        for i in tables:
            #TODO : don't calc empty minions
            for j in range(len(i), 8):
                i.append(np.array([-1] * self.CARD_SIZE))
            i[0] = self.observationCore(i[0])
            for j in range(1, len(i)):
                i[j] = self.observationMinion(i[j])
            buf = i[1:]
            #random.shuffle(buf)
            i[1:] = buf
        tables[0], tables[turn] = tables[turn], tables[0]
        tables_copy = copy.deepcopy(tables)

        for i in range(len(tables)):
            tables[i] = np.hstack(tuple(tables[i]))
            for j in range(len(tables_copy[i])):
                tables_copy[i][j] = tables_copy[i][j].tolist()
        return np.hstack(tuple(tables)), tables_copy

    def observationCore(self, core):
        return np.array(list(core))

    def observationPile(self, pile):
        pileCopy = copy.deepcopy(pile)
        #random.shuffle(pileCopy)
        for i in range(len(pileCopy)):
            pileCopy[i] = list(pileCopy[i])
            if pileCopy[i][1] == None:
                pileCopy[i][1] = np.array([-1] * self.CARD_SIZE)
            else:
                pileCopy[i][1] = observationMinion(pileCopy[i][1])
            pileCopy[i] = [pileCopy[i][0]] + pileCopy[i][1].tolist()
        return np.hstack(tuple(pileCopy)), pileCopy

    def observationHand(self, hand):
        handCopy = copy.deepcopy(hand)
        #random.shuffle(pileCopy)
        for i in range(len(hand[1]), 6):
            handCopy[1].append(None)
        for i in range(len(handCopy[1])):
            if handCopy[1][i] == None:
                handCopy[1][i] = np.array([-1] * self.CARD_SIZE)
            else:
                handCopy[1][i] = observationMinion(handCopy[1][i])
        return [handCopy[0]] + np.hstack(tuple(handCopy[1])), handCopy[1]

    def createStateObservation(self, state):

        observations = dict()
        observations["table"], tables = self.observationTable(state["battleGround"]["table"], state["turn"])
        observations["piles"] = []
        observations["hands"] = []
        pilesObs = []
        handsObs = []

        for i in state["piles"]:
            obs, obj = self.observationPile(i)
            observations["piles"].append(obj)
            pilesObs.append(obs)

        for i in state["hands"]:
            obs, obj = self.observationHand(i)
            observations["hands"].append(obj)
            handsObs.append(obs)
        observations["hands"][0], observations["hands"][state["turn"]] = observations["hands"][state["turn"]], observations["hands"][0]
        handsObs[0], handsObs[state["turn"]] = handsObs[state["turn"]], handsObs[0]
        handsObs = np.hstack(tuple(handsObs))

        observations["piles"][0], observations["piles"][state["turn"]] = observations["piles"][state["turn"]], observations["piles"][0]
        pilesObs[0], pilesObs[state["turn"]] = pilesObs[state["turn"]], pilesObs[0]
        pilesObs = np.hstack(tuple(pilesObs))

        observations["main"] = observations["table"].tolist() + pilesObs.tolist() + handsObs.tolist()

        cores = []
        units = []
        for i in tables:
            cores.append(i[0])
            units.append(i[1:])

        observations["cores"] = cores
        observations["units"] = units

        return observations

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
            obs_main = Variable(torch.Tensor([self.createStateObservation(n_obs)["main"]]))
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
    