#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os
import math
import sys

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

min_pts = sys.argv[1]
file_out = "/Databases/penguins"+str(min_pts)+".pickle"

penguins_metric,gold_count = pickle.load(open(base_directory+file_out,"rb"))
from scipy.stats.stats import pearsonr
steps = [10,20,30,40,50,60,70,80,90,100,110,120]
r =  {} #{s:[] for s in steps}
total = 0
less_than = {s: 0 for s in steps}
stopping = [5,10,15]

cumulative = []
#print gold_count
for st in steps:
    cumulative.append(len([g for g in gold_count if g <= st])/float(len(gold_count)))

plt.plot(steps,cumulative)
colors = ["red","green","grey"]
for ii,st in enumerate(stopping):
    r = {s:[] for s in steps}

    Y_temp = []
    for max_penguins in steps:
        Y_temp.append([])

        for image_index in range(len(gold_count)):
            if gold_count[image_index] > max_penguins:
                continue

            Y_temp[-1].append(penguins_metric[st][image_index])

    Y = [np.mean(y) for y in Y_temp]
    err = [np.std(y)/math.sqrt(len(y)) for y in Y_temp]

    plt.errorbar(steps,Y,err,color=colors[ii])

#plt.plot(steps,Y1)
#plt.plot(steps,Y2)
#print Y1
#plt.ylim(0.2,2.02)
plt.show()