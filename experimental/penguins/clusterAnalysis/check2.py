#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import urllib
import matplotlib.cbook as cbook
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import math

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveDBSCAN import DivisiveDBSCAN

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

print base_directory

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [5,10,15,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
#to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
import random
#for subject in collection2.find({"classification_count": 20}):

alreadyThere = True
user_markings = [] #{k:[] for k in steps}
user_ips = [] #{k:[] for k in steps}
zooniverse_id = "APZ0001vqf"
user_index = 0
for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
    user_index += 1
    if user_index == 21:
        break

    per_user = []

    ip = classification["user_ip"]
    try:
        markings_list = classification["annotations"][1]["value"]
        if isinstance(markings_list,dict):
            for marking in markings_list.values():
                if marking["value"] in ["adult","chick"]:
                    x,y = (float(marking["x"]),float(marking["y"]))

                    user_markings.append((x,y))
                    user_ips.append(ip)

    except (KeyError, ValueError):
            #classification["annotations"]
            user_index += -1



user_identified_penguins,clusters = DivisiveDBSCAN(3).fit(user_markings,user_ips,debug =True)#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")

#which users are in each cluster?
users_in_clusters = []
for c in clusters:
    users_in_clusters.append([])
    for p in c:
        i = user_markings.index(p)
        users_in_clusters[-1].append(user_ips[i])

X = []
Y = []
data = []
for i1 in range(len(user_identified_penguins)):
    for i2 in range(i1+1,len(user_identified_penguins)):
        #if i1 == i2:
        #    continue

        m1 = user_identified_penguins[i1]
        m2 = user_identified_penguins[i2]
        dist = math.sqrt((m1[0]-m2[0])**2+(m1[1]-m2[1])**2)
        X.append(dist)

        users1 = users_in_clusters[i1]
        users2 = users_in_clusters[i2]
        overlap = len([u for u in users1 if u in users2])
        Y.append(overlap)
        data.append((dist,overlap))

#plt.plot(X,Y,'.')
#plt.show()
data.sort(key = lambda x:x[0])
#data.sort(key = lambda x:x[1])

data2 = [overlap for dist,overlap in data]
#print data2.index(0)/float(len(data2))
print data2.index(1)/float(len(data2))