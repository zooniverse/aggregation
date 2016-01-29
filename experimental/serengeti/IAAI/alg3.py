#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

tauRange = range(0,101,10)
algPercent = []
currPercent = []

for tau in tauRange:
    print tau
    algPercent.append([])
    currPercent.append([])
    for j in range(5):
        print "--" + str(j)
        photos,users = setup(tau=tau)

        for p in photos.values():
            p.__sample__(5)
        for u in users.values():
            u.__prune__()

        #initialize things using majority voting
        for p in photos.values():
            p.__majorityVote__()

        #estimate the user's "correctness"
        for u in users.values():
            for s in speciesList:
                u.__speciesCorrect__(s,beta=0.2)

        for p in photos.values():
            p.__weightedMajorityVote__()

        correct = 0
        total = 0.
        for p in photos.values():
            if p.__goldStandardCompare__():
                correct += 1
            total += 1

        algPercent[-1].append(correct/total)


meanValues = [np.mean(p) for p in algPercent]
std = [np.std(p) for p in algPercent]
plt.errorbar(tauRange, meanValues, yerr=std)

plt.xlim((0,tauRange[-1]+1))
plt.show()

