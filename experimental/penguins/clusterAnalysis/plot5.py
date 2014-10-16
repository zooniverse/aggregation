#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os
import math

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

penguins_at,temp = pickle.load(open(base_directory+"/Databases/penguins_at_3_2.pickle","rb"))

from scipy.stats.stats import pearsonr
steps = [10,20,30,40,50,60,70,80,90,100,110,120]
r =  {} #{s:[] for s in steps}
total = 0
less_than = {s: 0 for s in steps}
stopping = [5,10,15]
for st in stopping:
    r = {s:[] for s in steps}
    total = 0
    less_than = {s: 0 for s in steps}

    for i in range(len(penguins_at[20])):
        if penguins_at[20][i] == 0:
            continue

        total += 1
        for s in steps:
            if penguins_at[20][i] > s:
                continue

            less_than[s] += 1
            r[s].append(penguins_at[st][i]/float(penguins_at[20][i]))

    Y1 = []
    Y2 = []
    err = []
    for s in steps:
        #print less_than[s]/float(total)
        Y1.append(less_than[s]/float(total))
        #print np.mean(r[s])
        Y2.append(np.mean(r[s]))
        err.append(np.std(r[s])/math.sqrt(len(r[s])))
        #print np.median(r[s])
        #print "===---"

    plt.errorbar(steps,Y2,err)

plt.plot(steps,Y1)
#plt.plot(steps,Y2)
print Y1
plt.ylim(0.2,1)
plt.show()