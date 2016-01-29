#!/usr/bin/env python
__author__ = 'greg'
#check to see what different tau values give us
from nodes import setup, speciesList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5]
#numUser = [25]
betas = [0.01,0.2,1,0]
betas = [0.2]
algPercent = {b:[] for b in betas}

f = open("/home/greg/Databases/betaValues.csv","wb")

for nn in numUser:
    print "== " + str(nn)
    for b in betas:
        algPercent[b].append([])

    for j in range(1):
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

                    f.write(str(u.userID) + ","+ str(s) +"," + str(u.speciesCorrect[s]) + "\n")
f.close()

