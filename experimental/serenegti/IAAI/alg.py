#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5,10,15,20,25]
algPercent = []
currPercent = []

for i in numUser:
    print i
    algPercent.append([])
    currPercent.append([])
    for j in range(10):
        photos,users = setup(tau=50)

        for p in photos.values():
            p.__sample__(i)
        for u in users.values():
            u.__prune__()

        #initialize things using majority voting
        for p in photos.values():
            p.__majorityVote__()

        #estimate the user's "correctness"
        for u in users.values():
            for s in speciesList:
                u.__speciesCorrect__(s,beta=0.03)

        for p in photos.values():
            p.__weightedMajorityVote__()

        correct = 0
        total = 0.
        for p in photos.values():
            if p.__goldStandardCompare__():
                correct += 1
            total += 1

        algPercent[-1].append(correct/total)

        for p in photos.values():
            p.__currAlg__()

        correct = 0
        total = 0.
        for p in photos.values():
            if p.__goldStandardCompare__():
                correct += 1
            total += 1

        currPercent[-1].append(correct/total)

meanValues = [np.mean(p) for p in algPercent]
std = [np.std(p) for p in algPercent]
plt.errorbar(numUser, meanValues, yerr=std)

meanValues = [np.mean(p) for p in currPercent]
std = [np.std(p) for p in currPercent]
plt.errorbar(numUser, meanValues, yerr=std)

plt.xlim((4,26))
plt.show()

