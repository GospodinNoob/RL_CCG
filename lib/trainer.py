import ccg
import utils

class Trainer():
    agents = []
    session = None
    n_actions = 71
    
    def __init__(self, session, agents):
        self.session = session
        self.agents = agents
        
    def playGame(self, session = None):
        self.playSteps(-1, session)
        return 
        
    def playSteps(self, n_steps, session = None):
        curSession = self.session
        if (session != None):
            curSession = session
        observation, validActions, validActionsEnv = self.session.processNewStateInfo()
        while not observation["end"] and n_steps != 0:
            turn = observation["turn"]
            curAgent = self.agents[turn]
            
            oldAdv = curSession.getHealthAdvantage(turn)
            
            n_steps -= 1
            action = curAgent.getAction(observation, validActions, validActionsEnv)
            n_observation, validActions, validActionsEnv = curSession.action(action)
            
            reward = oldAdv - curSession.getHealthAdvantage(turn)
            curAgent.record(0,
                            utils.createStateObservation(observation),
                            [int(k == action) for k in range(self.n_actions)], 
                            n_observation, 
                            reward, 
                            observation["end"])
            
            observation = n_observation
            
