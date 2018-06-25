'''
Created on 30 Aug 2016

@author: roxana
'''
import csv


def saveToCSV(results, config, exp, folder):
    columns = ['step', 'trial', 'reward']

    with open('results/data/' + folder + '/' + exp + '_' + config + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for step, res in enumerate(results):
            for trial, reward in enumerate(res):
                writer.writerow([step, trial, reward])


def readFromCSV(config, exp):
    with open('results/data/RND_SGM/' + exp + '_' + config + '.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        results = []
        for row in reader:
            step, _, reward = row
            step = int(step)
            # if step < 10000:
            reward = float(reward)
            if len(results) <= step:
                results.append([])
            results[step].append(reward)
    return results


def saveVariance(variance, config, exp):
    columns = ['config', 'trial', 'variance']

    with open('results/data/' + exp + '_' + config + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for trial, var in enumerate(variance):
            writer.writerow([config, trial, var])


def saveDistribution(distribution, config, exp, folder):
    columns = ['config', 'trial', 'resource', 'distribution']

    with open('results/data/' + folder + '/' + exp + '_' + config + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for trial, distr in enumerate(distribution):
            for res, value in enumerate(distr):
                writer.writerow([config, trial, res, value])
