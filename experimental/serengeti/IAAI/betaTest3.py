#!/usr/bin/env python
__author__ = 'greg'
#check to see what different tau values give us
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import csv

numUser = [5,10,15,20,25]
#numUser = [25]
betas = [0.01,0.2,1,0]
betas = [0.2]
algPercent = []



for nn in numUser:
    print "== " + str(nn)
    algPercent.append([])

    for j in range(10):
        print j
        photos,users = setup(tau=1)

        for p in photos.values():
            p.__sample__(nn)
        for u in users.values():
            u.__prune__()

        #initialize things using majority voting
        for p in photos.values():
            p.__majorityVote__()



        with open("/home/greg/Databases/betaValues.csv","rb") as f:
            reader = csv.reader(f)

            for userID,species,value in reader:
                value = float(value)

                users[userID].__setWeight__(species,value)

            f.close()


        for p in photos.values():
            p.__weightedMajorityVote__()

        correct = 0
        total = 0.
        for p in photos.values():
            if p.__goldStandardCompare__():
                correct += 1
            total += 1

        algPercent[-1].append(correct/total)

meanValues = [np.mean(p)*100 for p in algPercent]
print meanValues[-1]
std = [np.std(p)*100 for p in algPercent]

plt.errorbar(numUser,meanValues,yerr=std)

plt.xlabel("Number of Users per Photo")
plt.ylabel("Accuracy (%)")

plt.xlim((4,26))
plt.ylim((95,100))
plt.show()

