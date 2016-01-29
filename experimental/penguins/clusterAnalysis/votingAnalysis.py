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
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from clusterCompare import cluster_compare

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

penguins,temp = pickle.load(open(base_directory+"/Databases/penguins_vote.pickle","rb"))

#does this cluster have a corresponding cluster in the gold standard data?
#ie. does this cluster represent an actual penguin?

for image_index in range(len(penguins[5])):
    print len(penguins[5])
    user_penguins = penguins[5][0]
    gold_penguins = penguins[5][1]

    print len(user_penguins)
    print len(user_penguins[0])

    print "==="
    print len(user_penguins)
    for upen in user_penguins:
        print upen
        print cluster_compare(gold_penguins,[upen,])

    break