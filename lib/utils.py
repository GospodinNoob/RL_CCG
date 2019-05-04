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
    #print(len(np.hstack(tuple(observations["table"])).tolist()), len(pilesObs.tolist()), len(handsObs.tolist()))
    return observations

import random
import operator

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]