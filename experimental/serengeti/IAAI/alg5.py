#!/usr/bin/env python
__author__ = 'greg'
#check to see what different tau values give us
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5,10,15,20,25]
algPercent = []
tauRange = np.arange(1,33,4)

for tau in tauRange:
    print tau
    algPercent.append([])
    for j in range(10):
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
                u.__speciesCorrect__(s)

        for p in photos.values():
            p.__weightedMajorityVote__()

        goWithMostLikely = [p.__gowithMostLikely__() for p in photos.values()]
        goWithMostLikely = [a for a in goWithMostLikely if not(a is None)]
        algPercent[-1].append(np.mean(goWithMostLikely))

meanValues = [np.mean(p) for p in algPercent]
std = [np.std(p) for p in algPercent]

plt.errorbar(tauRange,meanValues,yerr=std)
plt.show()

