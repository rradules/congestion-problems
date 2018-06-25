'''
Created on 24 Nov 2016

@author: roxana
'''
from agents.multiAgent import MultiAgent


class Experiment():

    def __init__(self, task, multiAgent):
        assert isinstance(multiAgent, MultiAgent), "task should be MultiAgent."
        self.agent = multiAgent
        self.task = task

    def _interact(self):
        self.task.env.performAction(self.agent.getJointAction())  # action
        reward = self.task.getReward()
        self.agent.giveJointReward(reward)  # reward
        self.agent.integrateObservation(self.task.getObservation())  # next  state
        self.agent.performUpdates()  # update Q-values
        self.agent.newTurn()
        self.stepid += 1
        return reward

    def doTrials(self, number=1):
        distribution = []
        for dummy in range(number):
            self.agent.reset()
            self.stepid = 0
            self.task.reset()
            self.agent.initObservation(self.task.getObservation())
            while not self.task.isFinished():
                self._interact()
            distribution.append(self.task.env.computeDistribution())
            print 'Trial', dummy
