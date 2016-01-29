#!/usr/bin/env python
__author__ = 'greghines'
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import cPickle as pickle
import shutil
import urllib



client = pymongo.MongoClient()
db = client['penguin_2014-09-18']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

images = {}

pts = {}
userCount = {}
errorCount = 0
total = 0
at_5 = {}
at_10 = {}

center_5 = {}
center_10 = {}



step_1 = 5
step_2 = 8

toSkip = ["APZ0000ifx","APZ00014t4"]
mainSubject = "APZ0002uw3" #APZ0001jre
toPlot = None
numClassifications = {}


for r in collection.find():
    subject_id = r["subjects"][0]["zooniverse_id"]
    total += 1

    #if subject_id != mainSubject: #in toSkip:
    #    continue


    if not(subject_id in pts):
        pts[subject_id] = []
        userCount[subject_id] = 0
        numClassifications[subject_id] = []

    userCount[subject_id] += 1
    animalsPresent = r["annotations"][0]["value"] == "yes"
    #print animalsPresent
    if animalsPresent:
        c = 0
        for marking_index in r["annotations"][1]["value"]:
            try:
                marking = r["annotations"][1]["value"][marking_index]
                if marking["value"] == "adult":
                    pts[subject_id].append((float(marking["x"]),float(marking["y"])))
                    c += 1
            except TypeError:
                errorCount += 1
                userCount[subject_id] += -1
                break
            except ValueError:
                errorCount += 1
                continue

        if userCount[subject_id] <= step_1:
            numClassifications[subject_id].append(c)
    elif userCount[subject_id] <= step_1:
        numClassifications[subject_id].append(0)



    if userCount[subject_id] in [step_1,step_2]:
        X = np.array(pts[subject_id])
        if pts[subject_id] == []:
            continue
        db = DBSCAN(eps=20, min_samples=2).fit(X)
        labels = db.labels_

        #find out which points are noise
        class_member_mask = (labels == -1)


        X = np.array([pt for i,pt in enumerate(pts[subject_id]) if labels[i] != -1])
        centers = []
        if list(X) != []:

            af = AffinityPropagation(preference=-10).fit(X)
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_


            unique_labels = set(labels)



            for k in unique_labels:
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                if k == -1:
                    pass
                else:
                    xSet,ySet = zip(*list(X[class_member_mask]))
                    centers.append((np.mean(xSet),np.mean(ySet)))





        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #print "num penguins " + str(n_clusters_)

        if userCount[subject_id] == step_1:
            at_5[subject_id] = n_clusters_
            center_5[subject_id] = centers[:]
            toPlot = len(pts[mainSubject])

        else:
            at_10[subject_id] = n_clusters_
            center_10[subject_id] = centers[:]
            #mainSubject = subject_id

inBoth = [subject_id for subject_id in at_10 if (subject_id in at_5) and (at_10[subject_id] <= 20)]
# # print len(inBoth)
xValues = [at_5[subject_id] for subject_id in inBoth]
yValues = [at_10[subject_id] for subject_id in inBoth]
# print zip(inBoth,zip(x,y))
# plt.plot((0,100),(0,100),'--')
# # #print x
# # #print y
# plt.plot(x,y,'.')
# plt.show()

v = [np.var(numClassifications[subject_id]) for subject_id in inBoth]
r = [y - x for x,y in zip(xValues,yValues)]

for subject_id in inBoth:
    print subject_id, np.var(numClassifications[subject_id]), numClassifications[subject_id], (at_10[subject_id]- at_5[subject_id])

plt.plot(v,r,'.')
#plt.show()

