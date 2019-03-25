import numpy as np
import ccg
import copy

CARD_SIZE = None

def parse_state(states):
    #TODO : state to list states
    main = []
    our_units = []
    enemy_units = []
    enemy_core = []
    our_piles = []
    our_hand = []
    for st in states:
        turn = st["turn"]
        main.append(np.array(st["main"], dtype=np.float32)[None, None, :])
        our_units.append(np.array(st["units"][turn], dtype=np.float32)[:, None, :])
        enemy_units.append(np.array(st["units"][1 - turn], dtype=np.float32)[None, :, :])
        enemy_core.append(np.array([st["cores"][1 - turn]], dtype=np.float32)[None, :])
        our_piles.append(np.array(st["piles"][turn], dtype=np.float32)[:, None, :])
        our_hand.append(np.array(st["hands"][turn], dtype=np.float32)[None, :])
    return np.array(main), np.array(our_units), np.array(enemy_units), np.array(enemy_core), np.array(our_piles), np.array(our_hand)

def observationMinion(minion):
    state = list(minion)
    state[5] = int(state[5])
    return np.array(state)

def observationTable(table):
    global CARD_SIZE
    if CARD_SIZE == None:
        CARD_SIZE = len(observationMinion(ccg.Minion().getCurState()))
    tables = copy.deepcopy(table)
    for i in tables:
        #TODO : don't calc empty minions
        for j in range(len(i), 8):
            i.append(np.array([-1] * CARD_SIZE))
        i[0] = observationCore(i[0])
        for j in range(1, len(i)):
            i[j] = observationMinion(i[j])
        buf = i[1:]
        #random.shuffle(buf)
        i[1:] = buf
    tables_copy = copy.deepcopy(tables)

    for i in range(len(tables)):
        tables[i] = np.hstack(tuple(tables[i]))
        for j in range(len(tables_copy[i])):
            tables_copy[i][j] = tables_copy[i][j].tolist()
    return tables, tables_copy

def observationCore(core):
    return np.array(list(core))

def observationPile(pile):
    global CARD_SIZE
    if CARD_SIZE == None:
        CARD_SIZE = len(utils.observationMinion(ccg.Minion().getCurState()))
    pileCopy = copy.deepcopy(pile)
    #random.shuffle(pileCopy)
    for i in range(len(pileCopy)):
        pileCopy[i] = list(pileCopy[i])
        if pileCopy[i][1] == None:
            pileCopy[i][1] = np.array([-1] * CARD_SIZE)
        else:
            pileCopy[i][1] = observationMinion(pileCopy[i][1])
        pileCopy[i] = [pileCopy[i][0]] + pileCopy[i][1].tolist()
    return np.hstack(tuple(pileCopy)), pileCopy

def observationHand(hand):
    global CARD_SIZE
    if CARD_SIZE == None:
        CARD_SIZE = len(utils.observationMinion(ccg.Minion().getCurState()))
    handCopy = copy.deepcopy(hand)
    #random.shuffle(pileCopy)
    for i in range(len(hand[1]), 6):
        handCopy[1].append(None)
    for i in range(len(handCopy[1])):
        if handCopy[1][i] == None:
            handCopy[1][i] = np.array([-1] * CARD_SIZE)
        else:
            handCopy[1][i] = observationMinion(handCopy[1][i])
    return [handCopy[0]] + np.hstack(tuple(handCopy[1])), handCopy[1]

def createStateObservation(state):

    observations = dict()
    observations["table"], tables = observationTable(state["battleGround"]["table"])
    observations["piles"] = []
    observations["hands"] = []
    observations["turn"] = state["turn"]
    pilesObs = []
    handsObs = []

    for i in state["piles"]:
        obs, obj = observationPile(i)
        observations["piles"].append(obj)
        pilesObs.append(obs)

    for i in state["hands"]:
        obs, obj = observationHand(i)
        observations["hands"].append(obj)
        handsObs.append(obs)
    handsObs = np.hstack(tuple(handsObs))

    pilesObs = np.hstack(tuple(pilesObs))
    observations["main"] = np.hstack(tuple(observations["table"])).tolist() + pilesObs.tolist() + handsObs.tolist()

    cores = []
    units = []
    for i in tables:
        cores.append(i[0])
        units.append(i[1:])

    observations["cores"] = cores
    observations["units"] = units

    return observations