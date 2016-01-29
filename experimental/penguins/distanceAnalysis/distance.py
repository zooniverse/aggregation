#!/usr/bin/env python
#just compare the top and the bottom of the image
__author__ = 'greghines'
import numpy as np
import os
import sys
import cPickle as pickle
import math
import matplotlib.pyplot as plt
import pymongo
import urllib
import matplotlib.cbook as cbook

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
#from divisiveDBSCAN import DivisiveDBSCAN
#from divisiveDBSCAN_multi import DivisiveDBSCAN
#from clusterCompare import metric,metric2

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

penguins = pickle.load(open(base_directory+"/Databases/penguins_vote__.pickle","rb"))


lowest_cluster = float("inf")
highest_cluster = -float('inf')

#find the mid point
max_users = 20
y_values = []
num_images = 1
for ii,image in enumerate(penguins[max_users]):
    if num_images == ii:
        break
    #first - figure out what the average Y value was
    for cluster in image[1]:
        Y = np.mean(zip(*cluster[0])[1])

        y_values.append(Y)

    mid_point = np.mean(y_values)

    overall_dist = []

    low_clusters = []
    high_clusters = []
    overall_clusters = []

    low_dist = []
    high_dist = []

    #divide the points into low and high clusters
    for cluster in image[1]:
        X = np.mean(zip(*cluster[0])[0])
        Y = np.mean(zip(*cluster[0])[1])

        #REMEMBER - image is flipped!! (no idea why)
        if Y > mid_point:
            low_clusters.append((X,Y))
        else:
            high_clusters.append((X,Y))

        overall_clusters.append((X,Y))

    for c_1 in low_clusters:
        closest_neighbours = []
        #for j in range(i+1,len(low_clusters)):
        for c_2 in overall_clusters:
            if c_1 == c_2:
                continue

            dist = math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2)
            closest_neighbours.append(dist)

        #plt.plot((c_1[0],closest[0]),(c_1[1],closest[1]),color="blue")
        if closest_neighbours == []:
            assert(len(low_clusters) == 1)
        else:
            closest_neighbours.sort()
            low_dist.append(np.mean(closest_neighbours[0:1]))


    for c_1 in high_clusters:
        closest_neighbours = []
        #for j in range(i+1,len(low_clusters)):
        for c_2 in overall_clusters:
            if c_1 == c_2:
                continue

            dist = math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2)
            closest_neighbours.append(dist)

        #plt.plot((c_1[0],closest[0]),(c_1[1],closest[1]),color="blue")
        if closest_neighbours == []:
            assert(len(low_clusters) == 1)
        else:
            closest_neighbours.sort()
            high_dist.append(np.mean(closest_neighbours[0:1]))



    print np.mean(low_dist)
    print np.median(low_dist)
    print len(low_dist)

    n, bins, patches  =plt.hist(low_dist, 1000, normed=1,histtype='step', cumulative=True,color="green")
    print n[3],bins[3]
    print "====="
    print np.mean(high_dist)
    print np.median(high_dist)
    n, bins, patches  =plt.hist(high_dist, 1000, normed=1,histtype='step', cumulative=True,color="blue")
    print len(high_dist)
    plt.xlim(0,200)
    plt.ylim(0,0.4)
    plt.show()

