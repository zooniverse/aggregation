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
db = client['penguin_2014-09-19']
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

toSkip = ["APZ0002uw3","APZ0002jc3"]
mainSubject = "APZ0000dp1" #APZ0001jre
toPlot = None
numClassifications = []


for r in collection.find():
    subject_id = r["subjects"][0]["zooniverse_id"]
    total += 1

    if subject_id in toSkip:
        continue


    if not(subject_id in pts):
        pts[subject_id] = []
        userCount[subject_id] = 0

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

        numClassifications.append(c)



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

            af = AffinityPropagation(preference=-1500).fit(X)
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_

            try:
                unique_labels = set(labels)
            except TypeError:
                print pts[subject_id]
                userCount[subject_id] += -1
                continue


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
            #toPlot = len(pts[mainSubject])

        else:
            at_10[subject_id] = n_clusters_
            center_10[subject_id] = centers[:]
            mainSubject = subject_id
            break

# inBoth = [subject_id for subject_id in at_10 if (subject_id in at_5)]
# # print len(inBoth)
# x = [at_5[subject_id] for subject_id in inBoth]
# y = [at_10[subject_id] for subject_id in inBoth]
# print zip(inBoth,zip(x,y))
# plt.plot((0,100),(0,100),'--')
# # #print x
# # #print y
# plt.plot(x,y,'.')
# plt.show()

#print userCount
#print numClassifications

print mainSubject
r2 = collection2.find_one({"zooniverse_id":mainSubject})
url = r2["location"]["standard"]

if not(os.path.isfile("/home/greg/Databases/penguins/images/"+mainSubject+".JPG")):
    urllib.urlretrieve (url, "/home/greg/Databases/penguins/images/"+mainSubject+".JPG")

image_file = cbook.get_sample_data("/home/greg/Databases/penguins/images/"+mainSubject+".JPG")
image = plt.imread(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)

x,y = zip(*pts[mainSubject][0:sum(numClassifications)])
plt.plot(x,y,'.',color='blue')

x,y = zip(*center_5[mainSubject])
plt.plot(x,y,'.',color='red')
x,y = zip(*center_10[mainSubject])
plt.plot(x,y,'.',color='green')
plt.show()