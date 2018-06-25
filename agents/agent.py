'''
Created on 22 Feb 2017

@author: roxana
'''

import numpy as np


class Agent(object):
    init_exploration = 0.1  # aka epsilon
    exploration_decay = 0.9999  # per episode

    # flags for exploration strategies
    epsilonGreedy = True
    greedy = False

    def __init__(self, learner, num_features, num_actions, num_agents, index):
        self.learner = learner
        self.indexOfAgent = index
        self.lastaction = None
        self.state = []
        self.nextState = []

    def initEnv(self, startState):
        self.state = np.copy(startState)
        self.nextState = np.copy(startState)

    def _selectAction(self, state):
        if self.greedy:
            return self.learner._greedyPolicy(state)
        elif self.epsilonGreedy:
            temp = np.random.uniform(0., 1.)
            if temp < self._expl_proportion:
                return np.random.randint(0, len(self.learner._qValues(state)))
            else:
                return self.learner._greedyAction(state)

    def returnAction(self):
        return self.lastaction

    def getAction(self, potential):
        self.lastaction = self._selectAction(self.state)
        self.nextAction = None
        return np.array([self.lastaction])

    def getNextAction(self):
        self.nextAction = self.learner._greedyAction(self.state)
        return np.array([self.nextAction])

    def integrateObservation(self, obs):
        self.state = self.nextState
        self.nextState = obs

    def performUpdate(self, potential=-1):
        if potential != -1:
            self.learner.updatePotential(potential)
        self.learner._updateValues(self.state, self.lastaction, self.lastreward, self.nextState)

    def giveReward(self, rew):
        self.lastreward = rew

    def reset(self):
        self._expl_proportion = self.init_exploration
        self.learner.reset()

    def newTurn(self):
        self._expl_proportion *= self.exploration_decay
        self.learner.newTurn()
