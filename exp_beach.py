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
from utils.saveToCSV import saveToCSV  # , saveDistribution

if __name__ == '__main__':
    noAgents = 100
    episodes = 4000
    timesteps = 5
    trials = 10
    comment = "exp1_"
    # beaches = ['L', 'G', 'D', 'RA_1_1_2', 'RA_1_2_1','RA_1_2_3', 'RA_1_2_2']
    # beaches = ['RA_2+1+3'] #'RA_3+3','RA_4+2', 'RA_2+2+2','RA_5+1','RA_3+2+1','RA_1+3+2', 'RA_2+1+3']

    # 'L', 'G', 'D','RA_3+3','RA_4+2', 'RA_2+2+2','RA_5+1','RA_3+2+1','RA_1+3+2', 'RA_2+1+3']
    beaches = ['L', 'G', 'D', 'RA_3+3', 'RA_4+2', 'RA_2+1+3']
    # beaches = ['L', 'G', 'D', 'RA_3+3', 'RA_2+2+2','RA_6']
    # beaches = ['RA_6']
    # beaches = ['L', 'G', 'D', 'RA_weights', 'RA_1+8']
    # typeRes='orig' 'lin' 'pol' 'exp' 'gen'

    typeR = 'beach'
    tasks = [ResourceAllocation(b, noAgents, episodes, timesteps, typeR, braess=False, accident=False) for b in beaches]
    results = []
    features = 6

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

        saveToCSV(results, b, '00_' + comment + typeR, 'beach')

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

    pp = PdfPages('results/plots/beach/00_' + comment + typeR + '.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
