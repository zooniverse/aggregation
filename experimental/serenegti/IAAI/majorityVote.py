#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup
import numpy as np
import matplotlib.pyplot as plt

MVpercent = []
currPercent = []
numUser = [5,10,15,20,25]


for i in numUser:
    print i
    MVpercent.append([])
    currPercent.append([])

    for j in range(10):
        photos,users = setup()

        for p in photos.values():
            p.__sample__(i)
        for u in users.values():
            u.__prune__()

        for p in photos.values():
            p.__majorityVote__()

        correct = 0
        total = 0.
        for p in photos.values():
            if p.__goldStandardCompare__():
                correct += 1
            total += 1

        MVpercent[-1].append(correct/total)

        #repeat with the current algorithm
        ######
        ######
        for p in photos.values():
            p.__currAlg__()

        correct = 0
        total = 0.
        for p in photos.values():
            if p.__goldStandardCompare__():
                correct += 1
            total += 1

        currPercent[-1].append(correct/total)

meanValues = [np.mean(p) for p in MVpercent]
std = [np.std(p) for p in MVpercent]
plt.errorbar(numUser, meanValues, yerr=std)

meanValues = [np.mean(p) for p in currPercent]
std = [np.std(p) for p in currPercent]
plt.errorbar(numUser, meanValues, yerr=std)

plt.xlim((4,26))
plt.show()


