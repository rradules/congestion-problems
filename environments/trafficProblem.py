'''
Created on 27 Jul 2016

@author: roxana
'''
import numpy as np
from resources.resources import Resource, ResourceGroup


class trafficEnvironment(object):

    def __init__(self, agents, resources):
        self.sections = []
        self.noAgents = agents
        self.number = resources.number
        self.resource = resources
        self.noActions = 3
        self.sections = np.random.randint(resources.number, size=self.noAgents)
        # np.zeros(self.noAgents, dtype=int)
        self.init_sections = np.copy(self.sections)
        self.availableActions = [-1, 0, 1]

    def reset_resources(self, resources):
        self.resource = resources

    def _convertState(self):
        states = []
        for s in self.sections:
            state = [0 for _ in range(self.number)]
            state[s] += 1
            states.extend([state])
        return states

    def getSensors(self):
        return self._convertState()

    def reinitPositions(self):
        self.sections = np.copy(self.init_sections)

    def performAction(self, action):
        # action = action - np.ones(len(action))
        self.reset()
        for i, a in enumerate(action):
            state = self.sections[i] + self.availableActions[a]
            if state < 0: state = 0
            if state > self.resource.number - 1: state = self.resource.number - 1
            self.sections[i] = state
            self.resource.performAction(state)

    def reset(self):
        self.resource.reset()

    def getJointReward(self):
        rewards = np.array([self.resource.getReward(s) for s in self.sections])
        return rewards

    def computeVariance(self):
        sect_rew = self.resource.get_local_rewards()
        rew_array = sum([[sect_rew[i][1]] * sect_rew[i][2] for i in range(self.number)], [])
        var = np.var(rew_array)
        return var

    def computeDistribution(self):
        sect_rew = self.resource.get_local_rewards()
        return [sect_rew[i][2] for i in range(self.number)]


class TrafficLanes(object):

    def __init__(self, weights, capacities, groups, typeReward):
        assert (len(weights) == len(capacities)), "Weights and capacities should be of equal length!"
        resources = []
        self.groups = []
        self.number = len(weights)
        for i, (w, c) in enumerate(zip(weights, capacities)):
            resources.append(Resource(i, w, c, typeReward))
        for group in groups:
            self.groups.append(ResourceGroup(filter(lambda r: r.id in group, resources), typeReward))

    def reset(self):
        for res in self.groups:
            res.reset()

    def simulate_accident(self, lanes):
        w_l = [l - 1 if l - 1 > -1 else l + 1 for l in lanes]
        for i, l in enumerate(lanes):
            r_w = self.get_res_from_group(w_l[i])
            r = self.get_res_from_group(l)
            r.capacity = r.capacity / 2
            prev = r.weight
            r.weight = r_w.weight
            r_w.weight = prev

    def get_res_from_group(self, num):
        for g in self.groups:
            r = g.get_resource(num)
            if r: return r

    def performAction(self, section):
        for res in self.groups:
            r = res.get_resource(section)
            if r:  res.add_load(section)

    def getReward(self, section):
        reward = 0
        for res in self.groups:
            r = res.get_resource(section)
            if r: reward = res.getReward(r)
        return reward

    def globalReward(self):
        return sum(res.total_reward() for res in self.groups)

    def getConsumption(self):
        return sum([g.consumption for g in self.groups])

    def printResource(self):
        for group in self.groups:
            group.printResource()

    def get_local_rewards(self):
        sect_rew = [[] for _ in range(self.number)]
        for group in self.groups:
            for res in group.resources:
                sect_rew[res.id] = [res.id, res.localReward(), res.consumption]
        return sect_rew


class Resource_CaP_G_D(object):

    def __init__(self, weights, capacities, config, typeReward):
        assert (len(weights) == len(capacities)), "Weights and capacities should be of equal length!"
        self.resources = []
        self.config = config
        self.number = len(weights)
        for i, (w, c) in enumerate(zip(weights, capacities)):
            self.resources.append(Resource(i, w, c, typeReward))

    def reset(self):
        for res in self.resources:
            res.reset()

    def performAction(self, section):
        self.resources[section].add_load(1)

    def _get_resource(self, num):
        res = filter(lambda r: r.id == num, self.resources)
        return res[0] if res else None

    def simulate_accident(self, lanes):
        for l in lanes:
            self.resources[l].capacity = self.resources[l].capacity / 2
            prev = self.resources[l].weight
            if l - 1 > -1:
                self.resources[l].weight = self.resources[l - 1].weight
                self.resources[l - 1].weight = prev
            else:
                self.resources[l].weight = self.resources[l + 1].weight
                self.resources[l + 1].weight = prev

    def getReward(self, section):
        reward = 0
        if self.config == 'G':
            reward = sum(res.localReward() for res in self.resources)
        else:
            res = self._get_resource(section)
            if self.config == 'D':
                reward = res.differenceReward()
            elif self.config == 'L':
                reward = res.localReward()
            elif self.config == 'G+CaP':
                rew = [res.localReward() for res in self.resources]
                g = sum(rew)
                gi = res.localReward(sim=1)
                reward = [g, g - rew[section] + gi]
            elif self.config == 'L+CaP':
                rew = [res.localReward() for res in self.resources]
                l = res.localReward()
                li = res.localReward(sim=1)
                reward = [l, sum(rew) - l + li]
        return reward

    def globalReward(self):
        return sum(res.localReward() for res in self.resources)

    def getConsumption(self):
        return sum([g.consumption for g in self.resources])

    def printResource(self):
        for res in self.resources:
            res.printResource()

    def get_local_rewards(self):
        sect_rew = [[] for _ in range(self.number)]
        for res in self.resources:
            sect_rew[res.id] = [res.id, res.localReward(), res.consumption]
        return sect_rew
