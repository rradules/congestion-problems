'''
Created on 27 Jul 2016

@author: roxana
'''
from copy import deepcopy

import numpy as np


class BraessParadox():

    def __init__(self, agents):
        if agents == None:
            self.noAgents = 4000
        else:
            self.noAgents = agents
        self.resource = Graph()
        self.noActions = 3
        self.availableActions = [x for x in range(self.noActions)]
        self.actions = np.random.randint(self.noActions, size=self.noAgents)
        self.init_sections = self.actions

    def _convertState(self):
        states = []
        for s in self.actions:
            state = [0 for _ in range(3)]
            state[s] += 1
            states.extend([state])
        return states

    def reinitPositions(self):
        self.actions = self.init_sections

    def getSensors(self):
        state = self._convertState()
        return state  # the actions reflect the route taken (i.e, the state)

    def performAction(self, action):
        # compute outcomes
        self.actions = action
        self.resource.reset_load()
        for i in action:
            self.resource.add_load(i)

    def reset(self):
        self.resource = Graph()

    def getJointReward(self):
        travelTimes = self.resource.getRewards()
        rewards = [-travelTimes[a] for a in self.actions]
        return rewards


class Graph:
    def __init__(self):
        self.edge_dict = {}
        self.load = 0
        self.num_edges = 0
        # init with standard case
        self.add_edge('A', 'B', -1, 0)
        self.add_edge('B', 'C', 0.0, 0)
        self.add_edge('A', 'C', 45.0, 0)
        self.add_edge('B', 'D', 45.0, 0)
        self.add_edge('C', 'D', -1, 0)

    def __iter__(self):
        return iter(self.edge_dict.values())

    def add_edge(self, node1, node2, weight=-1, load=0):
        self.num_edges = self.num_edges + 1
        new_edge = Edge(node1, node2, weight, load)
        self.edge_dict[(node1, node2)] = new_edge
        return new_edge

    def get_edge(self, n1, n2):
        if (n1, n2) in self.edge_dict:
            return self.edge_dict[(n1, n2)]
        else:
            return None

    def get_edges(self):
        return self.edge_dict.keys()

    # can be generalized, I manually return the 3 possible paths
    def get_paths(self):
        return [[self.get_edge('A', 'B'), self.get_edge('B', 'D')],
                [self.get_edge('A', 'B'), self.get_edge('B', 'C'),
                 self.get_edge('C', 'D')],
                [self.get_edge('A', 'C'), self.get_edge('C', 'D')]]

    # get travel time or predicted travel time for a path
    def get_path_cost(self, path, predict):
        cost = 0
        for edge in path:
            cost += edge.get_travel_time(predict)
        return cost

    # get an array with travel time/path
    def getRewards(self):
        time = []
        for path in self.get_paths():
            time.append(self.get_path_cost(path, False))
        return time

    # get load of a path
    def get_load(self, path):
        load = 0
        for edge in path:
            load += edge.get_load()
        return load

    # reset loads
    def reset_load(self):
        self.load = 0
        for edge in self:
            edge.reset_load()

    # get average travel time per agent for entire network
    def globalReward(self):
        cost = 0
        for edge in self:
            cost += edge.get_load() * edge.get_travel_time(False)
        return cost / self.load

    # predict agent impact over entire network average travel time
    def predict_cost(self, path):
        cost = 0
        for edge in self:
            if (edge in path):
                cost += (1 + edge.get_load()) * edge.get_travel_time(True)
            else:
                cost += edge.get_load() * edge.get_travel_time(False)
        return cost / (self.load + 1)

    def printResource(self):
        for edge in self:
            print(edge)

    def print_path(self, path):
        for edge in path:
            print(edge)
        print('')

    def add_load(self, pathId):
        path = self.get_paths()[pathId]
        self.load += 1
        for edge in path:
            edge.add_load()
        # self.print_path(path)

    # return a new graph simulating removal of agent with the path
    def sim_remove_load(self, pathId):
        sim_graph = deepcopy(self)
        path = sim_graph.get_paths()[pathId]
        sim_graph.load -= 1
        for edge in path:
            edge.remove_load()
        return sim_graph


class Edge:
    """
    Edge in the network modeled in terms of the two composing nodes, 
    the traveling time (weight) and the load (traffic present on the edge)
    """

    def __init__(self, node1, node2, weight, load):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.load = load

    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2)

    def reset_load(self):
        # reset loads
        self.load = 0

    def get_nodes(self):
        return (self.node1, self.node2)

    # get traveling time for the edge
    def get_travel_time(self, predict):
        time = 0
        if (self.weight == -1):
            time = (predict + self.load) / 100.0
        else:
            time = self.weight
        return time

    def __str__(self):
        return (str(self.node1) + '-' + str(self.node2) + ': '
                + str(self.get_travel_time(False)) + ', ' + str(self.load))

    def get_load(self):
        return self.load

    def add_load(self):
        self.load += 1

    def remove_load(self):
        self.load -= 1
