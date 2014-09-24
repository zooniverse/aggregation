#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import urllib
import matplotlib.cbook as cbook

sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
import agglomerativeClustering


client = pymongo.MongoClient()
db = client['condor_2014-09-19']
collection = db["condor_classifications"]

i = 0
condor_pts = {}
condors_per_user = {}
classification_count = {}

check1 = 5
condors_at_1 = {}
check2 = 10
condors_at_2 = {}

total = 0
maxDiff = {}

check = [{} for i in range(0,11)]

for r in collection.find({"$and" : [{"tutorial":False}, {"subjects": {"$ne": []}} ]}):
    #zooniverse_id = r["zooniverse_id"]
    user_ip = r["user_ip"]
    zooniverse_id =  r["subjects"][0]["zooniverse_id"]


    if not(zooniverse_id in condor_pts):
        condor_pts[zooniverse_id] = set()
        #condor_user_id[zooniverse_id] = []
        classification_count[zooniverse_id] = 0
        condors_per_user[zooniverse_id] = []

    classification_count[zooniverse_id] += 1
    condor_count = 0
    if "marks" in r["annotations"][-1]:
        markings = r["annotations"][-1].values()[0]

        for marking_index in markings:
            marking = markings[marking_index]
            try:
                if marking["animal"] == "condor":
                    scale = 1.875
                    x = scale*float(marking["x"])
                    y = scale*float(marking["y"])
                    condor_pts[zooniverse_id].add(((x,y),user_ip))
                    #condor_user_id[zooniverse_id].append(user_ip)
                    condor_count += 1
            except KeyError:
                continue

    condors_per_user[zooniverse_id].append(condor_count)

    #if (classification_count[zooniverse_id] == 5) and (condor_pts[zooniverse_id] != []):
    if (np.mean(condors_per_user[zooniverse_id]) > 2) and (len(condors_per_user[zooniverse_id]) > 4):
        if condor_pts[zooniverse_id] != set([]):
            object_id = str(r["subjects"][0]["id"])
            url = r["subjects"][0]["location"]["standard"]
            cluster_center = agglomerativeClustering.agglomerativeClustering(condor_pts[zooniverse_id])
            break

if not(os.path.isfile("/home/greg/Databases/condors/images/"+object_id+".JPG")):
    urllib.urlretrieve (url, "/home/greg/Databases/condors/images/"+object_id+".JPG")

image_file = cbook.get_sample_data("/home/greg/Databases/condors/images/"+object_id+".JPG")
image = plt.imread(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)
#plt.show()
#
if cluster_center != []:
    x,y = zip(*cluster_center)
    plt.plot(x,y,'.',color='blue')

plt.show()

