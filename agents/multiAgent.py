'''
Created on 23 Aug 2016

@author: roxana
'''
import numpy as np


class MultiAgent():
    '''
    This class defines set of agents. 
    Each agent should be instance of IndexableAgent or its subclass.
    '''
    agentSet = []

    def __init__(self, potential=None):
        self.potential = potential
        self.agentSet = []

    def integrateObservation(self, obs):
        for index in range(len(self.agentSet)):
            self.agentSet[index].integrateObservation(obs[index])

    def initObservation(self, obs):
        for index in range(len(self.agentSet)):
            self.agentSet[index].initEnv(obs[index])

    def getJointAction(self):
        jointAction = np.zeros(len(self.agentSet), dtype=np.int)
        for index in range(len(self.agentSet)):
            jointAction[index] = self.agentSet[index].getAction(self.potential)
        if self.potential:
            self.potential.setConsumption(sum(jointAction))

        ''' 
        temp = self.agentSet[0]
        print 'State'
        print temp.state
        print 'Values'
        print [temp.learner._qValue(temp.state, i) for i in range(7)]
        '''
        return jointAction

    def returnActions(self):
        jointAction = np.zeros(len(self.agentSet), dtype=np.int)
        for index in range(len(self.agentSet)):
            jointAction[index] = self.agentSet[index].returnAction()
        return jointAction

    def giveJointReward(self, r):
        for index in range(len(self.agentSet)):
            self.agentSet[index].giveReward(r[index])

    def getNextAction(self):
        jointAction = np.zeros(len(self.agentSet), dtype=np.int)
        for index in range(len(self.agentSet)):
            jointAction[index] = self.agentSet[index].getNextAction()
        return jointAction

    def performUpdates(self, potentials=None):
        if potentials:
            for index in range(len(self.agentSet)):
                self.agentSet[index].performUpdate(potentials[index])
        else:
            for index in range(len(self.agentSet)):
                self.agentSet[index].performUpdate()

    def reset(self):
        for agent in self.agentSet:
            agent.reset()

    def newTurn(self):
        for agent in self.agentSet:
            agent.newTurn()

    def addAgent(self, agent):
        assert agent.indexOfAgent is not None, "Index should be identified"
        if len(self.agentSet) == 0:
            assert agent.indexOfAgent == 0, "Illegal indexing."
        else:
            ind = 0
            for elem in self.agentSet:
                assert ind == (elem.indexOfAgent), "Illegal indexing."
                ind += 1
            assert agent.indexOfAgent == ind, "Illegal indexing."
        self.agentSet.append(agent)

    def popAgent(self, index):
        agent = self.agentSet.pop(index)
        agent.setIndexOfAgent(None)


class MultiAgentPBRS(MultiAgent):

    def __init__(self, agents):
        MultiAgent.__init__(self)
        self.potentials = np.zeros(agents)

    def giveJointReward(self, r):
        """ give joint-reward to all agents.
            :key r: joint reward
            :type r: numpy array of doubles
        """

        rewards = r[:, 0]
        newPotentials = r[:, 1]

        for index in range(len(self.agentSet)):
            if self.agentSet[index].getProperty()["requireJointReward"]:
                gamma = self.agentSet[index].learner.rewardDiscount
                f = gamma * newPotentials - self.potentials
                self.agentSet[index].giveReward(rewards + f)
            else:
                gamma = self.agentSet[index].learner.rewardDiscount
                f = gamma * newPotentials[index] - self.potentials[index]
                self.agentSet[index].giveReward(rewards[index] + f)
        self.potentials = newPotentials
