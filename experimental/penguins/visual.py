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

    zooniverseID = r["subjects"][0]["zooniverse_id"]

    if zooniverseID != "APZ00003h1":
        continue

    print r

    for a in r["annotations"]:
        if ('value' in a) and not(a["value"]  in ["penguin", "adult", "no", "yes", "finished", "unfinished", "cant_tell", "", "chick", "eggs", "other"]):

            penguinsX.append([])
            penguinsY.append([])
            numAnnotations =  int(max(a["value"].keys(), key = lambda x:int(x)))

            for index in range(numAnnotations):
                penguinsX[-1].append(float(a["value"][str(index)]['x']))
                penguinsY[-1].append(float(a["value"][str(index)]['y']))




image_file = cbook.get_sample_data("/home/greg/Databases/penguins/images/5385f9cf7b9f994b1e0031f7.JPG")
image = plt.imread(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)
plt.plot(penguinsX[2],penguinsY[2],'o-')

plt.show()