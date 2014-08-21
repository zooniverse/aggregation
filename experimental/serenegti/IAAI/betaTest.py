#!/usr/bin/env python
__author__ = 'greg'
#check to see what different tau values give us
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5,10,15,20,25]
betas = [0,0.01,0.2,1]
algPercent = {b:[] for b in betas}

for nn in numUser:
    print nn
    for b in betas:
        algPercent[b].append([])

    for j in range(10):
        photos,users = setup(tau=10)

        for p in photos.values():
            p.__sample__(nn)
        for u in users.values():
            u.__prune__()

        #initialize things using majority voting
        for p in photos.values():
            p.__majorityVote__()

        for b in betas:

            #estimate the user's "correctness"
            for u in users.values():
                for s in speciesList:
                    u.__speciesCorrect__(s,b)

            for p in photos.values():
                p.__weightedMajorityVote__()

            correct = 0
            total = 0.
            for p in photos.values():
                if p.__goldStandardCompare__():
                    correct += 1
                total += 1

            algPercent[b][-1].append(correct/total)

p = []
for b in betas:
    meanValues = [np.mean(p) for p in algPercent[b]]
    print meanValues[-1]
    std = [np.std(p) for p in algPercent[b]]

    p.append(plt.errorbar(numUser,meanValues,yerr=std)[0])

plt.legend( ([str(b) for b in betas]), loc='lower right')
plt.xlim((4,32))
plt.show()

