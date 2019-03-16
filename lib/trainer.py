import ccg

class Trainer():
    agents = []
    session = None
    
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
        observation, validActionsEnv, validActions = self.session.processNewStateInfo()
        while not observation["end"] and n_steps != 0:
            n_steps -= 1
            action = self.agents[observation["turn"]].getAction(observation, validActionsEnv, validActions)
            observation, validActionsEnv, validActions = curSession.action(action)