'''
Created on 01 Aug 2016

@author: roxana
'''
import numpy as np


class Resource(object):

    def __init__(self, num, weight, capacity, typeExp):
        self.id = num
        self.weight = weight
        self.capacity = capacity
        self.consumption = 0
        self.typeExp = typeExp

    def add_load(self, load):
        self.consumption += load

    def remove_load(self):
        self.consumption -= 1

    def get_load(self):
        return self.consumption

    def reset(self):
        self.consumption = 0

    def isOverused(self):
        return self.capacity < self.consumption

    def differenceReward(self):
        local = self.localReward()
        dif = self.localReward(sim=1)
        return local - dif

    def localReward(self, sim=0):
        reward = 0
        if self.weight == 0:
            return reward
        cons = float(self.consumption - sim)
        if self.typeExp == 'beach':
            reward = cons * np.exp(-(cons) / self.capacity)
        elif self.typeExp == 'traffic':
            if self.isOverused():
                reward = self.weight * np.exp((-cons) / self.capacity)
            else:
                reward = self.weight * np.exp(-1)
        return reward

    def printResource(self):
        print 'Resource: ', self.id, ' - load/capacity: ', self.consumption, '/', self.capacity, ' - weight: ', self.weight


class ResourceGroup(Resource):

    def __init__(self, resources, typeR):
        self.typeR = typeR
        self.resources = resources
        self.weight = np.mean([r.weight for r in resources])
        self.capacity = sum(r.capacity for r in resources)
        self.consumption = 0

    def get_resource(self, num):
        res = filter(lambda r: r.id == num, self.resources)
        return res[0] if res else None

    def add_load(self, num):
        self.get_resource(num).add_load(1)
        self.consumption += 1

    def remove_load(self, num):
        self.get_resource(num).remove_load()
        self.consumption -= 1

    def get_load(self):
        return sum(r.consumption for r in self.resources)

    def reset(self):
        self.consumption = 0
        for r in self.resources:
            r.reset()

    def group_reward(self):
        if self.consumption == 0:
            self.consumption = self.get_load()
        assert (self.consumption == self.get_load()), "Abstract group consumption is not correct!"
        if self.typeR == 'beach':
            return - self.consumption * np.exp(float(-self.consumption) / self.capacity)
        else:
            if self.consumption <= self.capacity:
                return - self.weight * np.exp(-1)
            else:
                return - self.weight * np.exp(float(-self.consumption) / self.capacity)

    def getReward(self, res):
        if res.isOverused():
            return self.group_reward()
        else:
            return res.localReward()

    def total_reward(self):
        return sum(res.localReward() for res in self.resources)

    def printResource(self):
        for r in self.resources:
            r.printResource()


class Path(Resource):

    def __init__(self, num, path):
        self.id = num
        self.path = path
        self.weight = np.mean([r.weight for r in path])
        self.capacity = min([r.capacity for r in path])
        self.consumption = 0

    def add_load(self, num):
        self.consumption += num
        for edge in self.path:
            edge.add_load(num)

    def remove_load(self):
        self.consumption -= 1
        for edge in self.path:
            edge.remove_load()

    def get_load(self):
        return self.consumption

    def reset(self):
        self.consumption = 0
        for edge in self.path:
            edge.reset()

    def isOverused(self):
        return any(edge.isOverused() for edge in self.path)

    def differenceReward(self):
        return sum(edge.differenceReward() for edge in self.path)

    def localReward(self, sim=0):
        reward = sum(edge.localReward(sim) for edge in self.path)
        return reward

    def printResource(self):
        for el in self.path:
            el.printResource()
