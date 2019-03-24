import numpy as np
import copy

class BaseInfo:
    owner = -1
    keyName = 0

class Card(BaseInfo):
    cost = 0
    
    def __init__(self):
        pass
    
    def __init__(self, objList=None):
        pass
    
    def setOwner(self, owner):
        self.owner = owner
    
class Unit:
    wasActive = True
    maxHealth = 0
    curHealth = 0
    armour = 0
    damage = 0
    
    def dealDamageFrom(self, attacker):
        self.curHealth -= attacker.damage - self.armour
        attacker.activate()
        
    def isAlive(self):
        return self.curHealth > 0
    
    def activate(self):
        self.wasActive = True
        
    def isActive(self):
        return not self.wasActive
    
    def newTurn(self):
        self.wasActive = False
    
class Core(Unit):
    maxMana = 0
    curMana = 0
    
    def setOwner(self, owner):
        self.owner = owner
    
    def getCurState(self, playerNum, visible):
        return (self.damage, self.armour, self.curHealth, self.maxHealth, self.curMana, self.maxMana)
    
    def __init__(self, objList):
        self.damage, self.armour, self.maxHealth  = objList[2:]
        self.keyName = objList[0]
        self.curHealth = self.maxHealth
        self.maxMana = 1
        self.curMana = self.maxMana  
        self.active = True
        
    def spendMana(self, cost):
        self.curMana -= cost
        
    def newTurn(self):
        super(Core, self).newTurn()
        self.maxMana += 1
        self.maxMana %= 10
        self.curMana = self.maxMana 
        

    

class Minion(Card, Unit):
            
    def getCurState(self, playerNum = -1, visible = False):
        if(self.owner == playerNum) or visible:
            return (self.cost, self.damage, self.armour, self.curHealth, self.maxHealth, self.wasActive)
        return None
    
    def __init__(self, objList = None):
        if(objList == None):
            self.active = True
            return
        self.damage, self.armour, self.maxHealth, self.cost  = objList[2:]
        self.keyName = objList[0]
        self.curHealth = self.maxHealth
        self.active = False
        
class Hand(BaseInfo):
    cards = []
    maxSize = 6
    
    def __init__(self):
        self.cards = []
        maxSize = 6
        
    def addCard(self, card):
        self.cards.append(card)
    
    def isEmpty(self):
        return len(self.cards) == 0
    
    def isFull(self):
        return len(self.cards) == self.maxSize
    
    def GetCard(self, ind):
        card = self.cards[ind]
        del self.cards[ind]
        return card
    
    def getCurState(self, playerNum):
        cards_obs = []
        for card in self.cards:
            cards_obs.append(card.getCurState(playerNum, False))
        return (len(self.cards), cards_obs)
    
    def getValidActions(self, curMana):
        validActions = []
        for i, card in enumerate(self.cards):
            if(card.cost <= curMana):
                validActions.append(("play_hand", i))
        return validActions
    
class Pile(BaseInfo):
    cards = []
    
    def __init__(self):
        self.cards = []
        
    def __init__(self, pileCardsList):
        self.cards = copy.deepcopy(pileCardsList)
    
    def __init__(self, cardsList, pileSize):
        self.cards = np.random.choice(cardsList, pileSize).tolist()
        
    def setOwner(self, owner):
        for i in self.cards:
            i.setOwner(owner)
        
    def addCard(self, card):
        self.cards.append(card)
        
    def shuffle(self):
        self.cards = shuffle(self.cards)
        
    def isEmpty(self):
        return len(self.cards) == 0
    
    def GetCard(self):
        if(self.isEmpty()):
            return None
        card = self.cards[0]
        self.cards = self.cards[1:]
        return card
    
    def Top(self):
        if(self.isEmpty()):
            return None
        return self.cards[0]
    
    def getCurState(self, playerNum):
        top = self.Top()
        if top == None:
            return (0, None)
        return (len(self.cards), self.Top().getCurState(playerNum, False))
    
    def getValidActions(self, curMana, pileId):
        validActions = []
        top = self.Top()
        if top != None and top.cost <= curMana:
            validActions.append(("play", pileId))
        return validActions
    
class Table(BaseInfo):
    tables = None
    maxMinions = 7
    
    def __init__(self, cores):
        self.tables = [[copy.deepcopy(i)] for i in cores]
        
    def Attack(self, attacker, target):
        attackerUnit = self.tables[attacker[0]][attacker[1]]
        targetUnit = self.tables[target[0]][target[1]]
        targetUnit.dealDamageFrom(attackerUnit)
        attackerUnit.dealDamageFrom(targetUnit)
        
        if(not attackerUnit.isAlive() and (attacker[1] != 0)):
            del self.tables[attacker[0]][attacker[1]]
            
        if(not targetUnit.isAlive() and (target[1] != 0)):
            del self.tables[target[0]][target[1]]
                                                                    
    def getCurState(self, playerNum):
        state = []
        for i, table in enumerate(self.tables):
            state.append([])
            for j in table:
                state[i].append(j.getCurState(playerNum, True))
        return state
    
    def isEnd(self):
        for i, table in enumerate(self.tables):
            if(not table[0].isAlive()):
                return i
        return -1
    
    def isFull(self, playerNum):
        return len(self.tables[playerNum]) >= self.maxMinions + 1
    
    def Play(self, playerNum, card):
        if(card == None):
            return
        self.tables[playerNum][0].spendMana(card.cost)
        self.tables[playerNum].append(card)
    
    def getCurMana(self, playerNum):
        return self.tables[playerNum][0].curMana
    
    def newTurn(self, playerNum):
        for i in self.tables[playerNum]:
            i.newTurn()
        
    def getUnitsNum(self, playerNum):
        return len(self.tables[playerNum]) - 1
    
    def getHealthAdvantage(self, playerId):
        advantage = 0
        for i, table in enumerate(self.tables):
            if (i == playerId):
                advantage += table[0].curHealth
            else:
                advantage -= table[0].curHealth
        return advantage
    
    def getValidActions(self, playerNum):
        validActions = []
        for unit in range(1, len(self.tables[playerNum])):
            if self.tables[playerNum][unit].isActive():
                for i in range(len(self.tables)):
                    if (i != playerNum):
                        for j in range(len(self.tables[i])):
                            validActions.append(("attack", [playerNum, unit], [i, j]))
        return validActions

class Grave:
    graves = []
    
    def __init__(self, playersNum):
        self.graves = [[] for _ in range(playersNum)]
        
    def addMinion(self, playerNum, minion):
        self.graves[playerNum].append(minion)
        
    def getCurState(self, playerNum):
        state = []
        for i, grave in enumerate(self.graves):
            state.append([])
            for j in grave:
                state[i].append(j.getCurState(playerNum, True))
        return state
        
class BattleGround:
    table = None
    grave = None
    
    def __init__(self, cores):
        self.table = Table(cores)
        self.grave = Grave(len(cores))
        
    def getCurState(self, playerNum):
        state = dict()
        state["grave"] = self.grave.getCurState(playerNum)
        state["table"] = self.table.getCurState(playerNum)
        return state
    
    def Attack(self, attacker, target):
        self.table.Attack(attacker, target)
        
    def Play(self, playerNum, card):
        self.table.Play(playerNum, card)
        
    def isEnd(self):
        return self.table.isEnd()
    
    def isFull(self, playerNum):
        return self.table.isFull(playerNum)
    
    def getCurMana(self, playerNum):
        return self.table.getCurMana(playerNum)
    
    def getValidActions(self, playerNum):
        return self.table.getValidActions(playerNum)
    
    def newTurn(self, playerNum):
        self.table.newTurn(playerNum)
        
    def getUnitsNum(self, playerNum):
        return self.table.getUnitsNum(playerNum)
    
    def getHealthAdvantage(self, playerId):
        return self.table.getHealthAdvantage(playerId)
    
        
    
class Deck():
    core = None
    piles = []
    
    def __init__(self, core, piles, owner):
        self.core = copy.deepcopy(core)
        self.piles = copy.deepcopy(piles)
        self.core.setOwner(owner)
        for i in self.piles:
            i.setOwner(owner)
        
class Session:
    battleGround = None
    piles = []
    hands = []
    turn = 0
    globalTurn = 0
    playersNum = 0
    observation = None
    validActionsEnv = None
    validActions = None
    
    decks = None
    
    def __init__(self, decks):
        self.decks = copy.deepcopy(decks)
        self.init(decks)
    
    def init(self, decks):
        playersNum = len(decks)
        self.piles = [decks[i].piles.copy() for i in range(playersNum)]
        self.battleGround = BattleGround([decks[i].core for i in range(playersNum)])
        self.hands = [Hand() for _ in range(playersNum)]
        self.turn = 0
        self.globalTurn = 0
        self.playersNum = playersNum
        
    def envActionFromAction(self, action):
        #skip
        env_action = ("skip")
        if action == 0:
            return env_action

        #attack core
        action -= 1
        if(action < 7):
            env_action = ("attack", [self.turn, action], [1 - self.turn, 0])
            return env_action

        #attack unit
        action -= 7
        if(action < 49):
            env_action = ("attack", [self.turn, action // 7], [1 - self.turn, action % 7])
            return env_action

        #play from pile
        action -= 49
        if(action < 4):
            env_action = ("play", action)
            return env_action

        #play from hand
        action -= 4
        if(action < 6):
            env_action = ("play_hand", action)
            return env_action

        #move from pile to hand
        action -= 6
        env_action = ("move", action)
        if action < 4:
            return env_action
        return None
    
    def actionFromEnvAction(self, envAction):
        #TODO : rework
        for i in range(71):
            if envAction == self.envActionFromAction(i):
                return i
        return -1
    
    def getHealthAdvantage(self, playerId):
        return self.battleGround.getHealthAdvantage(playerId)
        
    def reset(self):
        self.init(copy.deepcopy(self.decks))
        return self.processNewStateInfo()
    
    def getNextState(self, action):
        nextState = copy.deepcopy(self)
        return nextState.action(action)
    
    def processNewStateInfo(self):
        self.observation = self.processObservation()
        self.validActionsEnv = self.getValidActions()
        self.validActions = []
        for i in range(71):
            if self.envActionFromAction(i) in self.validActionsEnv:
                self.validActions.append(1)
            else:
                self.validActions.append(0)
        return self.observation, self.validActions, self.validActionsEnv
    
    def action(self, action):
        return self.envAction(self.envActionFromAction(action))
    
    def envAction(self, action):
        if(action[0] == "attack"):
            self.battleGround.Attack(action[1], action[2])
        elif(action[0] == "move"):
            self.hands[self.turn].addCard(self.piles[self.turn][action[1]].GetCard())
        elif(action[0] == "play_hand"):
            if(not self.battleGround.isFull(self.turn)):
                self.battleGround.Play(self.turn, self.hands[self.turn].GetCard(action[1]))
        elif(action[0] == "play"):
            if(not self.battleGround.isFull(self.turn)):
                self.battleGround.Play(self.turn, self.piles[self.turn][action[1]].GetCard())
        elif(action == "skip"):
            self.turn += 1
            self.turn %= self.playersNum
            if(self.turn == 0):
                self.globalTurn += 1
            self.battleGround.newTurn(self.turn)
        return self.processNewStateInfo()
    
    def processObservation(self):
        state = dict()
        state["battleGround"] = self.battleGround.getCurState(self.turn)
        state["piles"] = []
        for i, pile in enumerate(self.piles):
            state["piles"].append([j.getCurState(self.turn) for j in pile])
        state["hands"] = []
        for i, hand in enumerate(self.hands):
            state["hands"].append(hand.getCurState(self.turn))
        state["loser"] = self.battleGround.isEnd()
        state["turn"] = self.turn
        state["end"] = (state["loser"] != -1) or (self.globalTurn > 60)
        return state
    
    def getValidActions(self):
        curMana = self.battleGround.getCurMana(self.turn)
        unitsOnTable = self.battleGround.getUnitsNum(self.turn)
        validActions = [("skip")] + self.battleGround.getValidActions(self.turn)
        if(unitsOnTable < 7):
            for i, pile in enumerate(self.piles[self.turn]):
                validActions += pile.getValidActions(curMana, i)
            validActions += self.hands[self.turn].getValidActions(curMana)
        if(not self.hands[self.turn].isFull()):
            for i, pile in enumerate(self.piles[self.turn]):
                if(not pile.isEmpty()):
                    validActions.append(("move", i))
        return validActions
        
    
    
        