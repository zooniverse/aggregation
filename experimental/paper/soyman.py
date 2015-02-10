#!/usr/bin/env python
__author__ = 'greg'
import pymongo
from aggregation import base_directory
from penguinAggregation import PenguinAggregation
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
elif os.path.exists("/Users/greg"):
    sys.path.append("/Users/greg/Code/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from divisiveKmeans import DivisiveKmeans
from multiClickCorrect import __ibcc__2

clusterAlg = DivisiveKmeans().__fit__

penguin = PenguinAggregation()



client = pymongo.MongoClient()
db = client['penguin_2015-01-18']
collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

subjects = subject_collection.find({"metadata.path":{"$regex":"MAIVb2012a"}})
accuracy = []
num_gold =0
could_have = 0
missed = 0
false_pos = 0
overlaps = {}
#overlaps2 = []
Xt = []
Yt = []

gold_dict = {}

with open("/Users/greg/Databases/MAIVb2013_adult_RAW.csv","rb") as f:
    for lcount,(l,s) in enumerate(zip(f.readlines(),subjects)):
        if lcount == 20:
            break
        print lcount
        image_fname = l.split(",")[0]
        #print image_fname
        gold_string = l.split("\"")[1]
        gold_markings = gold_string[:-2].split(";")
        pts = [tuple(m.split(",")[:2]) for m in gold_markings]
        if len(pts) != len(list(set(pts))):
            print "Grrrrr"
        pts = list(set(pts))
        num_gold += len(pts)

        gold_dict[zooniverse_id] = pts[:]

        zooniverse_id = s["zooniverse_id"]

        width = s["metadata"]["original_size"]["width"]
        height = s["metadata"]["original_size"]["height"]

        pts = [(int(x)/(width/1000.),int(y)/(height/563.)) for (x,y) in pts]


        if penguin.__get_status__(zooniverse_id) != "complete":
            continue
        penguin.__readin_subject__(zooniverse_id)

        blankImage = penguin.__cluster_subject__(zooniverse_id, clusterAlg,fix_distinct_clusters=True)

penguin.__roc__()
#__ibcc__2(penguin.clusterResults,penguin.users_per_subject)

# plt.plot(Xt,Yt,'.')
# plt.xlabel("Large cluster size")
# plt.ylabel("Small cluster size")
# plt.xlim((min(Xt)-0.05,max(Xt)+0.05))
# plt.ylim((min(Yt)-0.05,max(Yt)+0.05))
# plt.show()
# for i in range(1,10):
#     print sum([1 for j in Yt if i == j])
# Y = []
# yErr = []
# X = []
# for i in range(1,10):
#     y = []
#     for a,j in overlaps.items():
#         if i == j:
#             acc = penguin.__user_accuracy__(a)
#             y.append(acc)
#
#     if y != []:
#         Y.append(np.median(y))
#         yErr.append(np.std(y))
#         X.append(i)
#
# plt.errorbar(X,Y,yerr=yErr)
# plt.xlim(0,X[-1]+1)
# plt.show()

# print false_pos,num_gold
# n,bins,patches = plt.hist(overlaps.values(),bins=max(overlaps.values()))
# print n
# plt.xlabel("Number of false positives found by a single user")
# plt.show()
# print could_have,missed
# print could_have/float(missed)
# print len(accuracy)
# print np.mean(accuracy)
# print np.median(accuracy)
#
# plt.plot(numGold,accuracy,'.')
# plt.xlabel("Gold Number of Penguins")
# plt.ylabel("Accuracy")
# plt.ylim((0.8,1.01))
# plt.show()