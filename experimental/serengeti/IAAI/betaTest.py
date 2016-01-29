#!/usr/bin/env python
__author__ = 'greg'
#check to see what different tau values give us
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5,10,15,20,25]
#numUser = [25]
betas = [0.01,0.2,1,0]
#betas = [0.4]
algPercent = {b:[] for b in betas}

for nn in numUser:
    print "== " + str(nn)
    for b in betas:
        algPercent[b].append([])

    for j in range(20):
        print j
        photos,users = setup(tau=1)

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
#print zip(zip(betas,["--","-","--","-",]),["grey","black","grey","black"])
for (b,fmt),color in zip(zip(betas,["--","--","-","-"]),["grey","black","grey","black"]):
    meanValues = [np.mean(p)*100 for p in algPercent[b]]
    print meanValues[-1]
    std = [np.std(p)*100 for p in algPercent[b]]

    p.append(plt.errorbar(numUser,meanValues,yerr=std,fmt=fmt,color=color)[0])

plt.xlabel("Number of Users per Photo")
plt.ylabel("Accuracy (%)")

plt.legend( ([str(b) for b in betas]), loc='lower right')
plt.xlim((4,26))
plt.ylim((93,100))
plt.show()

