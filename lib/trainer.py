import ccg
import utils
import numpy as np
import copy

def entropy(p):
    return -np.sum(np.exp(p) * p, -1)

class Trainer():
    agents = []
    session = None
    n_actions = 71
    
    def __init__(self, session, agents):
        self.session = session
        self.agents = agents
        
    def dropSession(self):
        if(self.session != None):
            self.session.reset()
        for i in self.agents:
            i.dropPlan()
        
    def playGame(self, session = None, record = False, replay_id = 0, evaluate = True, resetAfterGame = True):
        return self.playSteps(-1, session, record = record, replay_id = replay_id, evaluate = evaluate, resetAfterGame = resetAfterGame)
        
    def playSteps(self, n_steps, session = None, record = True, replay_id = 0, evaluate = False, resetAfterGame = True):
        curSession = self.session
        if (session != None):
            curSession = session
        if (n_steps < 0):
            curSession.reset()
            for i in self.agents:
                i.dropPlan()
        validActionsEnv = None
        validActions = None
        observation = None
            
        observation, validActions, validActionsEnv = curSession.processNewStateInfo()
        adv_before_skips = [0, 0]
        obs_before_skips = [None, None]
        act_before_skips = [0, 0]
        
        entropy_log = [[], []]
        
        while not observation["end"] and n_steps != 0:
            turn = observation["turn"]
            curAgent = self.agents[turn]
            
            oldAdv = curSession.getHealthAdvantage(turn)
            adv_before_skips[turn] = oldAdv
            
            n_steps -= 1
            observation, validActions, validActionsEnv = curSession.processNewStateInfo()
            action, log_probs = curAgent.getAction(observation, validActions, validActionsEnv, session = curSession, evaluate = evaluate)
            if(validActions[action] == 0):
                print("wtf")
                return
            n_observation, validActions, validActionsEnv = curSession.action(action)
            entropy_log[turn].append(entropy(log_probs))
            if (record):
                if(turn == n_observation["turn"]):
                    reward = curSession.getHealthAdvantage(turn) - oldAdv# + len(observation["battleGround"]["table"][turn]) - 1# - 0.1
                    curAgent.record(replay_id,
                                    utils.createStateObservation(observation),
                                    [int(k == action) for k in range(self.n_actions)], 
                                    n_observation, 
                                    reward, 
                                    observation["end"])
                else:
                    obs_before_skips[turn] = copy.deepcopy(observation)
                    act_before_skips[turn] = action

                    if(obs_before_skips[1 - turn] != None):
                        reward = curSession.getHealthAdvantage(1 - turn) - adv_before_skips[1 - turn]# - 0.1
                        self.agents[1 - turn].record(replay_id,
                                        utils.createStateObservation(obs_before_skips[1 - turn]),
                                        [int(k == act_before_skips[1 - turn]) for k in range(self.n_actions)], 
                                        n_observation, 
                                        reward, 
                                        n_observation["end"])
            
            observation = n_observation
        if (record):
            for i in self.agents:
                i.endRecord(replay_id)
        
        result = curSession.getGameStats()
        
        if(observation["end"] and resetAfterGame):
            curSession.reset()
            for i in self.agents:
                i.dropPlan()
        
        return result, entropy_log
            
    def train(self):
        losses = []
        for i in self.agents:
            losses.append(i.train())
            
        return losses
            
            
