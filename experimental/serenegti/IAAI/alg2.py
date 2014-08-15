#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
import random
import random

numUser = [5,10,15,20,25]
algPercent = []
currPercent = []

photos,users = setup()
photoIDs = photos.keys()
random.shuffle(photoIDs)

from itertools import izip, chain, repeat



def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]



crossValidate = list(chunks(photoIDs,len(photoIDs)/5))
leftover = crossValidate.pop(-1)
for i,v in enumerate(leftover):
    crossValidate[i].append(v)

for i in numUser:
    print i
    algPercent.append([])
    currPercent.append([])
    for j in range(len(crossValidate)):
        photos,users = setup()

        for photoID in crossValidate[j]:
            photos[photoID].__useGoldStandard__()

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
                u.__speciesCorrect__(s)

        for p in photos.values():
            p.__weightedMajorityVote__()

        correct = 0
        total = 0.
        for photoID,p in photos.items():
            if not(photoID in crossValidate[j]):
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


