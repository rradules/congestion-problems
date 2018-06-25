'''
Created on 23 Nov 2016

@author: roxana
'''

import numpy as np


class QLearner():
    initLearningRate = 0.1
    learningRateDecay = 0.9999

    randomInit = False

    gamma = 0.9  # 0.9 # aka gamma

    def __init__(self, num_features, num_actions, index, potential=None):
        self.indexOfAgent = index
        self.num_actions = num_actions
        self.num_features = num_features
        self.learningRate = self.initLearningRate
        self._initQtable()
        self.reset()
        self._initPotential()
        self.potential = potential

    def _initQtable(self):
        if self.randomInit:
            self._theta = np.random(self.num_actions, self.num_features) / 10.
        else:
            self._theta = np.zeros((self.num_actions, self.num_features))

    def _initPotential(self):
        self.potValue = [0, 0]

    def _qValues(self, state):
        return np.dot(self._theta, state)

    def _qValue(self, state, action):
        return np.dot(self._theta[action], state)

    def _greedyAction(self, state):
        temp = self._qValues(state)
        if self.potential:
            # print self.potential.cons
            temp = np.array([self.potential.computePotential(a) + temp[a] for a in range(len(temp))])
        return np.random.choice(np.where(temp == temp.max())[0])

    def _greedyPolicy(self, state):
        tmp = np.zeros(self.num_actions)
        tmp[self._greedyAction(state)] = 1
        return tmp

    def reset(self):
        self.learningRate = self.initLearningRate
        self._initQtable()
        self._initPotential()

    def newTurn(self):
        self.learningRate *= self.learningRateDecay
        self._initPotential()

    def updatePotential(self, pot):
        self.potValue[0] = self.potValue[1]
        self.potValue[1] = pot

    def _updateValues(self, state, action, reward, next_state):
        # shaping function
        f = self.gamma * self.potValue[1] - self.potValue[0]
        td_error = reward + f + self.gamma * max(self._qValues(next_state)) - self._qValue(state, action)
        self._theta[action] += np.dot(state, self.learningRate * td_error)

