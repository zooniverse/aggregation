#!/usr/bin/env python
import pymongo
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import urllib
import matplotlib.cbook as cbook
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.patches import Ellipse
from copy import deepcopy
__author__ = 'greghines'

client = pymongo.MongoClient()
db = client['penguins']
collection = db["penguin_classifications"]

penguins = {}
penguinType = {}
adults = {}
chicks = {}
eggs = {}
count = {}
fNames = {}
pCount = {}

i = 0
pen = 0
total = 0
for r in collection.find():
    for a in r["annotations"]:
        if ('value' in a) and not(a["value"]  in ["penguin", "adult", "no", "yes", "finished", "unfinished", "cant_tell", "", "chick", "eggs", "other"]):

            zooniverseID = r["subjects"][0]["zooniverse_id"]
            if not(zooniverseID in penguins):
                penguins[zooniverseID] = []
                penguinType[zooniverseID] = []
                count[zooniverseID] = 1
                url = r["subjects"][0]["location"]["standard"]
                fNames[zooniverseID] = url.split("/")[-1]
            else:
                count[zooniverseID] += 1

            #penguins[zooniverseID].append(len(a["value"]))

            for index in a["value"]:
                point = a["value"][index]

                penguins[zooniverseID].append((float(point["x"]),float(point["y"])))
                penguinType[zooniverseID].append(point["value"])


overallCount = {2:0,3:0,4:0,5:0}



diffs = []

for zooniverseID,c in count.items():
    if c >= 3:
        print zooniverseID
        X = np.array(penguins[zooniverseID])
        db = DBSCAN(eps=10, min_samples=2).fit(X)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_



        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #print('Estimated number of clusters: %d' % n_clusters_)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)

        for k in unique_labels:
            print k
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            t = [str(pType) for i,pType in enumerate(penguinType[zooniverseID]) if labels[i] == k]

            print t
            if k == -1:
                pass
            else:
                xSet,ySet = zip(*list(X[class_member_mask]))
                Mx = np.mean(xSet)
                My = np.mean(ySet)

                for x,y in zip(xSet,ySet):
                    diffs.append((x,y,math.sqrt((x-Mx)**2+(y-My)**2)))

        break