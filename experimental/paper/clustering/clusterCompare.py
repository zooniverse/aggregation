#!/usr/bin/env python
__author__ = 'ggdhines'
from penguinAggregation import PenguinAggregation
import random
import os
import sys
import cPickle as pickle
import aggregation
import matplotlib.pyplot as plt
import numpy as np

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from agglomerativeClustering import Ward
from divisiveKmeans import DivisiveKmeans

clusterAlg2 = DivisiveKmeans().__fit__
clusterAlg = Ward().__fit__

penguin = PenguinAggregation()
subject_ids = pickle.load(open(aggregation.base_directory+"/Databases/penguin_gold.pickle","rb"))


X1 = []
Y1 = []
X2 = []
Y2 = []
nonEmpty = 0
index = -1
random.shuffle(subject_ids)
while True:
    index += 1
#for i,subject in enumerate(random.sample(subject_ids,50)):
    #subject = "APZ000173v"
    subject = subject_ids[index]
    print nonEmpty,index

    penguin.__readin_subject__(subject,users_to_skip=["caitlin.black"])
    numMarkings,time_to_cluster = penguin.__cluster_subject__(subject, clusterAlg)
    if numMarkings == 0:
        continue

    nonEmpty += 1
    if nonEmpty == 30:
        break

    X1.append(numMarkings)
    Y1.append(time_to_cluster)

    numMarkings2,time_to_cluster = penguin.__cluster_subject__(subject, clusterAlg2,fix_distinct_clusters=True)
    X2.append(numMarkings2)
    Y2.append(time_to_cluster)

    print numMarkings,numMarkings2

# plt.plot(X1,Y1,"+",color="black",label="Agglomerative")
# plt.plot(X2,Y2,"o",color="black",label = "Divisive k-means")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Runtime of Clustering Algorithm")
# plt.legend(loc="upper left")
# print len(X1),len(X2)
# print np.mean(Y1)
# print np.mean(Y2)
#
# plt.show()