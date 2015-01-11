#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import random
import bisect

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveKmeans import DivisiveKmeans

results = {}

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc



client = pymongo.MongoClient()
db = client['plankton_2015-01-01']
classification_collection = db["plankton_classifications"]
subject_collection = db["plankton_subjects"]
user_collection = db["plankton_users"]

ip_listing = []

#the header for the csv input file
f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")

subject_list = []
user_list = []

subject_results = {}
for subject in subject_collection.find({"state":"complete"}):
    zooniverse_id = subject["zooniverse_id"]
    bisect.insort(subject_list,zooniverse_id)


# for user in user_collection.find():
#     try:
#         name = user["name"]
#         bisect.insort(user_list,name)
#     except KeyError:
#         print user
#         raise

for classification in classification_collection.find():
    if classification["subjects"] == []:
        continue

    if "user_name" in classification:
        user = classification["user_name"]
    else:
        user = classification["user_ip"]

    try:
        user_index = index(user_list,user)
    except ValueError:
        bisect.insort(user_list,user)

print "****"
for classification in classification_collection.find():
    if classification["subjects"] == []:
        continue

    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    try:
        subject_index = index(subject_list,zooniverse_id)
    except ValueError:
        continue

    if "user_name" in classification:
        user = classification["user_name"]
    else:
        user = classification["user_ip"]


    if not(zooniverse_id in subject_results):
        subject_results[zooniverse_id] = []


    try:
        user_index = index(user_list,user)
    except ValueError:
        print "not found"
        continue

    for marking in classification["annotations"]:
        if "started_at" in marking:
            break

        X = []
        Y = []
        for i in range(4):
            p = marking["p"+str(i)]
            X.append(float(p[0]))
            Y.append(float(p[1]))

        pt = (np.mean(X),np.mean(Y))
        try:
            species = marking["species"]
        except KeyError:
            species = "NA"
        subject_results[zooniverse_id].append((pt,species,user))


for counter,(zooniverse_id,markings) in enumerate(subject_results.items()):
    if markings == []:
        continue

    pts,species,users = zip(*markings)
    if len(pts) > 100:
        continue
    plankton,clusters,users_l = DivisiveKmeans(3).fit2(pts,users,debug=True)#,jpeg_file=base_directory+"/Databases/condors/images/"+object_id)

    for c in clusters:
        for m in c:
            index = pts.index(m)
            print species[index],users[index]

        print "--"

    if counter == 25:
        break

