#!/usr/bin/env python
import pymongo
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
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

penguinsX = []
penguinsY = []


i = 0
pen = 0
total = 0
for r in collection.find():

    if not("user_name" in r):
        continue
    #if r["user_name"] != "camallen":
    #    continue

    for a in r["annotations"]:
        if ('value' in a) and not(a["value"]  in ["penguin", "adult", "no", "yes", "finished", "unfinished", "cant_tell", "", "chick", "eggs", "other"]):
            numAnnotations =  int(max(a["value"].keys(), key = lambda x:int(x)))

            for index in range(numAnnotations):
                penguinsX.append(float(a["value"][str(index)]['x']))
                penguinsY.append(float(a["value"][str(index)]['y']))

heatmap, xedges, yedges = np.histogram2d(penguinsX, penguinsY, bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

fig = plt.clf()
plt.imshow(heatmap, extent=extent)
plt.colorbar()
plt.show()