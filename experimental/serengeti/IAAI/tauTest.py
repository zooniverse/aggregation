#!/usr/bin/env python
__author__ = 'greg'
#check to see what different tau values give us
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5,10,15,20,25]
tauRange = [1,10,100]
algPercent = {t:[] for t in tauRange}

for nn in numUser:
    print nn
    for t in tauRange:
        algPercent[t].append([])

    for j in range(10):
        print "== " + str(j)
        photos,users = setup(tau=10)

        for p in photos.values():
            p.__sample__(nn)
        for u in users.values():
            u.__prune__()

        #initialize things using majority voting
        for p in photos.values():
            p.__majorityVote__()

        for t in tauRange:


            #estimate the user's "correctness"
            for u in users.values():
                for s in speciesList:
                    u.__speciesCorrect__(s,beta=0.01)

            for p in photos.values():
                p.__weightedMajorityVote__(tau=t)

            correct = 0
            total = 0.
            for p in photos.values():
                if p.__goldStandardCompare__():
                    correct += 1
                total += 1

            algPercent[t][-1].append(correct/total)

p = []
for t in tauRange:
    meanValues = [np.mean(p) for p in algPercent[t]]
    std = [np.std(p) for p in algPercent[t]]

    plt.errorbar(numUser,meanValues,yerr=std)

plt.legend( tuple([str(t) for t in tauRange] ))
plt.xlim((4,32))
plt.show()
