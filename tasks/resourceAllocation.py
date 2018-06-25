'''
Created on 22 Feb 2017

@author: roxana
'''

import numpy as np

# from environments.BraessParadox import BraessParadox
from environments.RNDPath import RNDPath
from environments.RNDsgm import RND
from environments.trafficProblem import trafficEnvironment, TrafficLanes, Resource_CaP_G_D


class ResourceAllocation():

    def __init__(self, taskString, agents, episodes, timesteps, typeRes, braess=1, accident=False):
        self.isGameFinished = False
        self.accident = accident
        self.maximumEpisode = episodes
        self.maximumTimesteps = timesteps
        self.currentEp = 0
        self.currentTS = 0
        self.taskString = taskString
        self.typeRes = typeRes
        self.braess = braess

        if self.braess == 1:
            task = RND(agents, taskString, typeRes)
        elif self.braess == 2:
            task = RNDPath(agents, taskString, typeRes)
        else:
            resource = self._init_res(self.taskString, typeRes)
            task = trafficEnvironment(agents, resource)

        self.env = task
        self.globalRewards = [[] for _ in range(episodes)]

    def reset_resource(self):
        if self.braess:
            self.env.reset()
        else:
            self.env.reset_resources(self._init_res(self.taskString, self.typeRes))

    def _init_res(self, task, typeRes):

        lanes = 6
        capacities = lanes * np.ones(lanes)
        traffic = True
        # weights = [1, 5, 1, 1, 5, 1]
        weights = [1, 1, 1, 1, 1, 1]

        if task == "RA_3+3":
            res = beach3_3(weights, capacities, typeRes)
        elif task == "RA_4+2":
            res = beach4_2(weights, capacities, typeRes)
        elif task == "RA_5+1":
            res = beach5_1(weights, capacities, typeRes)
        elif task == "RA_1+3+2":
            res = beach1_3_2(weights, capacities, typeRes)
        elif task == "RA_2+1+3":
            res = beach2_1_3(weights, capacities, typeRes)
        elif task == "RA_2+2+2":
            res = beach2_2_2(weights, capacities, typeRes)
        elif task == "RA_3+2+1":
            res = beach3_2_1(weights, capacities, typeRes)
        elif task == "RA_6":
            res = beach1(weights, capacities, typeRes)
        elif task == 'G+CaP':
            res = Resource_CaP_G_D(np.ones(lanes), capacities, 'G+CaP', typeRes)
        elif task == 'L+CaP':
            res = Resource_CaP_G_D(np.ones(lanes), capacities, 'L+CaP', typeRes)
        elif task == 'D':
            if typeRes == 'beach':
                res = Resource_CaP_G_D(weights, capacities, 'D', typeRes)
            else:
                if traffic:
                    res = Resource_CaP_G_D([1, 5, 10, 1, 5, 10, 1, 5, 10], [167, 83, 33, 17, 9, 17, 33, 83, 167], 'D',
                                           typeRes)
                else:
                    res = Resource_CaP_G_D(weights, capacities, 'D', typeRes)
        elif task == 'L':
            if typeRes == 'beach':
                res = Resource_CaP_G_D(weights, capacities, 'L', typeRes)
            else:
                if traffic:
                    res = Resource_CaP_G_D([1, 5, 10, 1, 5, 10, 1, 5, 10], [167, 83, 33, 17, 9, 17, 33, 83, 167], 'L',
                                           typeRes)
                else:
                    res = Resource_CaP_G_D(weights, capacities, 'L', typeRes)
        elif task == 'G':
            if typeRes == 'beach':
                res = Resource_CaP_G_D(weights, capacities, 'G', typeRes)
            else:
                if traffic:
                    res = Resource_CaP_G_D([1, 5, 10, 1, 5, 10, 1, 5, 10], [167, 83, 33, 17, 9, 17, 33, 83, 167], 'G',
                                           typeRes)
                else:
                    res = Resource_CaP_G_D(weights, capacities, 'G', typeRes)
        elif task == "RA_weights":
            res = traffic_non_contig(typeRes)
        elif task == "RA_1+8":
            res = traffic_1_8(typeRes)
        else:
            res = traffic_1_8(typeRes)

        return res

    def reset(self):
        self.env.reset()
        self.reset_resource()
        self.isGameFinished = False
        self.currentEp = 0
        self.currentTS = 0

    def isFinished(self):
        return self.isGameFinished

    def getReward(self):
        jointReward = self.env.getJointReward()

        if self.currentTS >= self.maximumTimesteps:
            self.env.reinitPositions()
            self.currentTS = -1
            self.globalRewards[self.currentEp].append(self.env.resource.globalReward())
            self.currentEp += 1
        if self.currentEp >= self.maximumEpisode:
            self.isGameFinished = True
            self.env.resource.printResource()
            self.currentEp = 0
            self.currentTS = 0
        if self.accident and self.currentEp == 500 and self.currentTS == -1:
            print 'Simulating accident: '
            # self.env.resource.simulate_accident([2,8])
            # self.env.resource.simulate_accident([0,8])
            self.env.resource.printResource()
            print "\n"
            self.env.simulate_accident()
            self.env.resource.printResource()
            print "\n"

        self.currentTS += 1
        return jointReward

    def getObservation(self):
        return self.env.getSensors()


def traffic_1_8(typeRes):
    return TrafficLanes([1, 5, 10, 1, 5, 10, 1, 5, 10], [167, 83, 33, 17, 9, 17, 33, 83, 167],
                        [[0], [x for x in range(1, 9)]], typeRes)


def traffic_non_contig(typeRes):
    return TrafficLanes([1, 5, 10, 1, 5, 10, 1, 5, 10], [167, 83, 33, 17, 9, 17, 33, 83, 167],
                        [[0, 3, 6], [1, 4, 7], [2, 5, 8]], typeRes)


def beach1(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1, 2, 3, 4, 5]], typeRes)


def beach3_3(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1, 2], [3, 4, 5]], typeRes)


def beach4_2(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1, 2, 3], [4, 5]], typeRes)


def beach2_4(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1], [2, 3, 4, 5]], typeRes)


def beach5_1(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1, 2, 3, 4], [5]], typeRes)


def beach1_5(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0], [1, 2, 3, 4, 5]], typeRes)


def beach1_3_2(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0], [1, 2, 3], [4, 5]], typeRes)


def beach2_1_3(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1], [2], [3, 4, 5]], typeRes)


def beach2_2_2(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1], [2, 3], [4, 5]], typeRes)


def beach3_2_1(weights, capacities, typeRes):
    return TrafficLanes(weights, capacities, [[0, 1, 2], [3, 4], [5]], typeRes)
