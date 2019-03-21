import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ccg
import copy
import numpy as np

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

class ActorNetwork(nn.Module):
    def __init__(self, state, VEC_SIZE = 100, epsilon = 0.5):
        super().__init__()
        
        self.CARD_SIZE = len(self.observationMinion(ccg.Minion().getCurState()))
        state = self.createStateObservation(state)
        self.MAIN_SIZE = len(state["main"])
        self.CORE_SIZE = len(state["cores"][0])
        self.PILE_SIZE = len(state["piles"][0][0])
        self.HAND_SIZE = len(state["hands"][0])
        self.VEC_SIZE = VEC_SIZE
        
        self.epsilon = epsilon

        self.field2vec = nn.Sequential(
            nn.Linear(self.MAIN_SIZE, 512,), 
            nn.ELU(), 
            nn.Linear(512, VEC_SIZE))


        self.skip_qvalue = nn.Sequential(
            nn.Linear(self.MAIN_SIZE, 512,), 
            nn.ELU(), 
            nn.Linear(512, 1))

        self.card2vec = nn.Sequential(
            nn.Linear(self.CARD_SIZE, 512,), 
            nn.ELU(), 
            nn.Linear(512, VEC_SIZE))

        self.attack_units_qvalues = nn.Sequential(
            nn.Linear(self.VEC_SIZE, 512,),  #[field, attacker_card, target_card]
            nn.ELU(), 
            nn.Linear(512, 1))

        self.core2vec = nn.Sequential(
            nn.Linear(self.CORE_SIZE, 512,), 
            nn.ELU(), 
            nn.Linear(512, VEC_SIZE))

        self.pile2vec = nn.Sequential(
            nn.Linear(self.PILE_SIZE, 512,), 
            nn.ELU(), 
            nn.Linear(512, VEC_SIZE))

        self.hand2vec = nn.Sequential(
            nn.Linear(self.HAND_SIZE, 512,), 
            nn.ELU(), 
            nn.Linear(512, VEC_SIZE))

        self.attack_core_qvalues = nn.Sequential(
            nn.Linear(VEC_SIZE, 512,),  #[field, attacker_card, core]
            nn.ELU(), 
            nn.Linear(512, 1))

        self.play_card_qvalues = nn.Sequential(
            nn.Linear(VEC_SIZE, 512,),  #[field, pile]
            nn.ELU(), 
            nn.Linear(512, 1))

        self.play_hand_card_qvalues = nn.Sequential(
            nn.Linear(VEC_SIZE, 512,),  #[field, hand]
            nn.ELU(), 
            nn.Linear(512, 1))

        self.move_card_qvalues = nn.Sequential(
            nn.Linear(VEC_SIZE, 512,),  #[field, pile]
            nn.ELU(), 
            nn.Linear(512, 1))
        
        
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


    def parse_state(self, state):
        main = np.array(state["main"], dtype=np.float32)[None, None, None, :]
        our_units = np.array(state["units"][0], dtype=np.float32)[None, :, None, :]
        enemy_units = np.array(state["units"][1], dtype=np.float32)[None, None, :, :]
        enemy_core = np.array([state["cores"][1]], dtype=np.float32)[None, None, :]
        our_piles = np.array(state["piles"][0], dtype=np.float32)[None, :, None, :]
        our_hand = np.array(state["hands"][0], dtype=np.float32)[None, None, :]
        return main, our_units, enemy_units, enemy_core, our_piles, our_hand

    def get_qvalues_from_state(self, state):
        main, our_units, enemy_units, enemy_core, our_piles, our_hand = self.parse_state(state)
        main = np.array([main])
        enemy_units = np.array([enemy_units])
        enemy_core = np.array([enemy_core])
        our_piles = np.array([our_piles])
        our_units = np.array([our_units])
        our_hand = np.array([our_hand])
        return self.forward(main, our_units, enemy_units, enemy_core, our_piles, our_hand)

    def forward(self, main, our_units, enemy_units, enemy_core, our_piles, our_hand):

        qvalues = self.compute_qvalue(
            torch.from_numpy(main),
            torch.from_numpy(our_units),
            torch.from_numpy(enemy_units),
            torch.from_numpy(enemy_core),
            torch.from_numpy(our_piles),
            torch.from_numpy(our_hand)
        )

        return qvalues

    def compute_qvalue(self, field, card, target_card, core, pile, hand):
        field_vec = self.field2vec(field)
        card_vec = self.card2vec(card)
        target_vec = self.card2vec(target_card)
        pile_vec = self.pile2vec(pile)
        core_vec = self.core2vec(core)
        hand_vec = self.hand2vec(hand)

        batch_size = len(core)
        #print(hand_vec.shape)
        #print(field_vec.shape, card_vec.shape, target_vec.shape)
        attack_units_qvalue = self.attack_units_qvalues(field_vec + card_vec + target_vec)
        #print(attack_units_qvalue.shape)
        attack_units_qvalue = attack_units_qvalue.view(-1)
        attack_units_qvalue = attack_units_qvalue.view(batch_size, len(attack_units_qvalue) // batch_size)
        #print(attack_units_qvalue.shape[1] == 49)

        #print(field_vec.shape, card_vec.shape, core_vec.shape)
        attack_core_qvalue = self.attack_core_qvalues(field_vec + card_vec + core_vec)
        #print(attack_core_qvalue.shape)
        attack_core_qvalue = attack_core_qvalue.view(-1)
        attack_core_qvalue = attack_core_qvalue.view(batch_size, len(attack_core_qvalue) // batch_size)
        #print(attack_core_qvalue.shape[1] == 7)

        play_card_qvalue = self.play_card_qvalues(field_vec + pile_vec).view(-1)
        play_card_qvalue = play_card_qvalue.view(batch_size, len(play_card_qvalue) // batch_size)
        #print(play_card_qvalue.shape[1] == 4)

        skip_qvalue = self.skip_qvalue(field).view(-1)
        skip_qvalue = skip_qvalue.view(batch_size, len(skip_qvalue) // batch_size)
        #print(skip_qvalue.shape[1] == 1)

        play_hand_card_qvalue = self.play_hand_card_qvalues(field_vec + hand_vec).view(-1)
        play_hand_card_qvalue = play_hand_card_qvalue.view(batch_size, len(play_hand_card_qvalue) // batch_size)
        #print(play_hand_card_qvalue.shape[1])

        move_card_qvalue = self.move_card_qvalues(field_vec + pile_vec).view(-1)
        move_card_qvalue = move_card_qvalue.view(batch_size, len(move_card_qvalue) // batch_size)
        #print(move_card_qvalue.shape[1])

        return torch.cat((skip_qvalue, 
                          attack_core_qvalue, 
                          attack_units_qvalue, 
                          play_card_qvalue, 
                          play_hand_card_qvalue,
                          move_card_qvalue
                         ), dim=1)

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

