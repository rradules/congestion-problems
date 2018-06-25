'''
Created on 30 Aug 2016

@author: roxana
'''
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.font_manager import FontProperties
import pylab
from matplotlib.backends.backend_pdf import PdfPages

from agents.agent import Agent
from agents.multiAgent import MultiAgent
from experiment.experiment import Experiment
from learners.QLearner import QLearner
from tasks.resourceAllocation import ResourceAllocation
from utils.saveToCSV import saveToCSV

if __name__ == '__main__':
    noAgents = 500
    episodes = 10000
    timesteps = 8
    trials = 20
    comment = "exp1_TLD_accident08_"
    beaches = ['L']
    # beaches = ['G']
    # beaches = ['D']
    # beaches = ['RA_weights']
    # beaches = ['RA_1+8']
    # typeRes='orig' 'lin' 'pol' 'exp' 'gen'

    typeR = 'traffic'
    tasks = [ResourceAllocation(b, noAgents, episodes, timesteps, typeR, braess=3, accident=False) for b in beaches]
    results = []
    features = 9

    labels = []
    cm = pylab.get_cmap('viridis')
    colors = cm(np.linspace(0, 1, len(beaches)))
    # colors = ['red','green','blue', 'yellow', 'violet', 'cyan', 'yellowgreen', 'indigo', 'darkred', 'mediumslateblue','lightslategrey','darkorange','dimgray']

    for t, b in enumerate(beaches):
        print b
        task = tasks[t]
        ma = MultiAgent()

        print 'Creating agents...'
        for i in range(noAgents):
            learner = QLearner(features, task.env.noActions, i)
            agent = Agent(learner,
                          num_features=features,
                          num_actions=task.env.noActions,
                          num_agents=noAgents,
                          index=i)
            ma.addAgent(agent)

        exp = Experiment(task, ma)

        print 'Running experiment...'
        distribution = exp.doTrials(number=trials)
        #
        results = task.globalRewards

        saveToCSV(results, b, '00_' + comment + typeR, 'traffic')

        sigma = np.array([np.std(r) for r in results])
        mean = np.array([np.mean(r) for r in results])

        win = 1000 + 50 * t
        err = [sigma[i] for i in range(0, len(mean), win)]
        avg = [mean[i] for i in range(0, len(mean), win)]

        l, = plt.plot(mean, color=colors[t], label=b, alpha=0.8)
        labels.append(l)
        plt.errorbar([i for i in range(0, len(mean), win)], avg, fmt='o', color=colors[t], yerr=err)

    # fmts = ['s', 'o', '^]

    plt.ylabel('Global utility')
    plt.xlabel('Episodes')
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=labels, loc='center left', bbox_to_anchor=(1, 0.5))

    pp = PdfPages('results/plots/traffic/00_' + b + comment + typeR + '.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
