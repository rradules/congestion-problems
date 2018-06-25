'''
Created on 27 Jul 2016

@author: roxana
'''
import numpy as np
from pybrain.rl.environments import Environment

from resources.resources import Resource, Path, ResourceGroup


class RNDPath(Environment):

    def __init__(self, agents, config, typeRes):
        self.typeRes = typeRes
        self.config = config
        self.actionList = ['ABD', 'ABCD', 'ACD']
        if agents is None:
            self.noAgents = 50
        else:
            self.noAgents = agents
        if config == "[ABD],[ABCD,ACD]":
            self.resource = ResourceAbstractionGraph([['ABD'], ['ABCD', 'ACD']], typeRes)
        elif config == "[ABD,ABCD],[ACD]":
            self.resource = ResourceAbstractionGraph([['ABD', 'ABCD'], ['ACD']], typeRes)
        elif config == "[ABD,ACD],[ABCD]":
            self.resource = ResourceAbstractionGraph([['ABD', 'ACD'], ['ABCD']], typeRes)
        elif config == "[ABD],[ABCD],[ACD]":
            self.resource = ResourceAbstractionGraph([['ABD'], ['ABCD'], ['ACD']], typeRes)
        elif config == "[ABD,ABCD,ACD]":
            self.resource = ResourceAbstractionGraph([['ABD', 'ABCD', 'ACD']], typeRes)
        else:
            self.resource = Graph(config, typeRes)
            self.actionList = [0, 1, 2]
        self.noActions = 3
        self.availableActions = [x for x in range(self.noActions)]
        self._random_init()

    def _random_init(self):
        self.init_sections = np.random.randint(self.noActions, size=self.noAgents)
        self.actions = np.copy(self.init_sections)

    def _convertState(self):
        states = []
        for s in self.actions:
            state = [0 for _ in range(3)]
            state[s] += 1
            states.extend([state])
        return states

    def reinitPositions(self):
        self.actions = np.copy(self.init_sections)

    def getSensors(self):
        state = self._convertState()
        return state  # the actions reflect the route taken (i.e, the state)

    def performAction(self, action):
        self.actions = action
        self.resource.reset()
        for i in action:
            self.resource.performAction(self.actionList[i])

    def simulate_accident(self):
        self.resource.simulate_accident()

    def reset(self):
        self.resource.reset()
        self._random_init()

    def getJointReward(self):
        travelTimes = self.resource.getReward()
        rewards = np.array([travelTimes[a] for a in self.actions])
        return rewards

    def computeDistribution(self):
        return self.resource.get_res_cons()


class Graph(object):
    def __init__(self, config, typeRes):
        self.config = config
        self.typeRes = typeRes

        self.resources = build_resources(1, 5, typeRes)
        self.paths = get_paths(self.resources)

    # def getConsumption(self):
    #    return get_consumption(self.resources)

    def get_res_cons(self):
        cons = [p.consumption for p in self.paths]
        return cons

    def simulate_accident(self):
        # 'AC', 'BC', 'BD',
        get_edge('BC', self.resources).weight = 5
        get_edge('AC', self.resources).weight = 1
        get_edge('BD', self.resources).weight = 1

    def reverse_accident(self):
        get_edge('BC', self.resources).weight = 1
        get_edge('AC', self.resources).weight = 5
        get_edge('BD', self.resources).weight = 5

    def getReward(self):
        time = []
        if self.config == 'L':
            time = [path.localReward() for path in self.paths]
        elif self.config == 'G':
            g = self.globalReward()
            time = g * np.ones(len(self.paths))
        elif self.config == 'D':
            time = [path.differenceReward() for path in self.paths]
        elif self.config == 'G+CaP':
            cost = []
            cost_i = []
            for path in self.paths:
                cost.append(path.localReward())
                cost_i.append(path.localReward(sim=1))
            g = self.globalReward()
            time = [[g, g - cost[p] + cost_i[p]] for p in range(len(self.paths))]
        elif self.config == 'L+CaP':
            cost = []
            cost_i = []
            for path in self.paths:
                cost.append(path.localReward())
                cost_i.append(path.localReward(sim=1))
            g = self.globalReward()
            time = [[cost[p], g - cost[p] + cost_i[p]] for p in range(len(self.paths))]
        return time

    def reset(self):
        for path in self.paths:
            path.reset()

    def globalReward(self):
        globalR = sum(edge.localReward() for edge in self.resources)
        return globalR

    def printResource(self):
        for edge in self.resources:
            edge.printResource()

    def performAction(self, pathId):
        self.paths[pathId].add_load(1)


class ResourceAbstractionGraph(object):
    def __init__(self, groups, typeR):
        self.groupConfig = np.copy(groups)
        self.groups = []
        self.typeR = typeR

        self.resources = build_resources(1, 5, typeR)  # weight, capacity
        self.paths = get_paths(self.resources)

        for group in groups:
            self.groups.append(ResourceGroup(filter(lambda r: r.id in group, self.paths), typeR))

            # def getConsumption(self):

    #    return get_consumption(self.resources)

    def get_res_cons(self):
        cons = [p.consumption for p in self.paths]
        return cons

    def reset(self):
        for res in self.groups:
            res.reset()

    def simulate_accident(self):
        # 'AC', 'BC', 'BD',
        get_edge('BC', self.resources).weight = 5
        get_edge('AC', self.resources).weight = 1
        get_edge('BD', self.resources).weight = 1

    def reverse_accident(self):
        get_edge('BC', self.resources).weight = 1
        get_edge('AC', self.resources).weight = 5
        get_edge('BD', self.resources).weight = 5

    def performAction(self, section):
        for res in self.groups:
            r = res.get_resource(section)
            if r:  res.add_load(section)

    # get an array with travel time/path
    def getReward(self):
        reward = []
        for path in self.paths:
            for res in self.groups:
                r = res.get_resource(path.id)
                if r: reward.append(res.getReward(r))
        return reward

    # get average travel time per agent for entire network
    def globalReward(self):
        globalR = sum(edge.localReward() for edge in self.resources)
        return globalR

    def printResource(self):
        for edge in self.resources:
            edge.printResource()


def build_resources(weight, capacity, typeR):
    if typeR == 'beach':
        # edges = ['AB', 'AC', 'BD', 'CD']
        edges = ['AB', 'AC', 'BC', 'BD', 'CD']
        resources = [Resource(e, weight, capacity, typeR) for e in edges]
        # resources.append(Resource('BC', 0, 0))
        return resources
    else:
        weights = [1, 5, 1, 5, 1]
        capacities = [3, 7, 5, 7, 3]
        edges = ['AB', 'AC', 'BC', 'BD', 'CD']
        resources = [Resource(e, weights[i], capacities[i], typeR) for i, e in enumerate(edges)]
        return resources


def get_edge(num, resources):
    res = filter(lambda r: r.id == num, resources)
    return res[0] if res else None


def get_paths(resources):
    path_names = ['ABD', 'ABCD', 'ACD']
    paths = [[get_edge('AB', resources), get_edge('BD', resources)],
             [get_edge('AB', resources), get_edge('BC', resources), get_edge('CD', resources)],
             [get_edge('AC', resources), get_edge('CD', resources)]]
    return [Path(path_names[i], p) for i, p in enumerate(paths)]


def get_consumption(resources):
    return get_edge('BD', resources).consumption + get_edge('CD', resources).consumption
