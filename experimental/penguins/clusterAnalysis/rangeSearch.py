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

min_pts = [2,3,4,5]
penguins_metric = {}
gold_count = {}
for mm in min_pts:
    file_out = "/Databases/penguins"+str(mm)+".pickle"

    penguins_metric[mm],gold_count[mm] = pickle.load(open(base_directory+file_out,"rb"))

max_ = []

for num_users in [5,10,15]:
    Y = []
    for mm in min_pts:
        Y_temp = []
        results = penguins_metric[mm][num_users]
        gold_results = gold_count[mm]
        for image_index in range(len(gold_results)):
            if gold_results[image_index] <= 80:
                Y_temp.append(results[image_index])

        Y.append(np.mean(Y_temp))

    max_.append(max(Y))
    plt.plot(min_pts,Y,'-o')

plt.show()

plt.plot([5,10,15],max_,'-o')
plt.show()