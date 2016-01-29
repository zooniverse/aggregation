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

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveDBSCAN import DivisiveDBSCAN

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [5,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
import random
#for subject in collection2.find({"classification_count": 20}):
for zooniverse_id in random.sample(to_sample,10):
    subject = collection2.find_one({"zooniverse_id": zooniverse_id})
    subject_index += 1
    #if subject_index == 2:
    #    break
    #zooniverse_id = subject["zooniverse_id"]
    print "=== " + str(subject_index)
    print zooniverse_id

    alreadyThere = True
    user_markings = {k:[] for k in steps}
    user_ips = {k:[] for k in steps}

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
                        if not((x,y) in per_user):
                            per_user.append((x,y))
                            for s in steps:
                                if user_index < s:
                                    user_markings[s].append((x,y))
                                    user_ips[s].append(ip)

        except (KeyError, ValueError):
                #classification["annotations"]
                user_index += -1

    if user_markings[5] == []:
        print "skipping empty"
        subject_index += -1
        continue

    clusters = {}

    user_identified_penguins,clusters[5] = DivisiveDBSCAN(3).fit(user_markings[5],user_ips[5],debug=True)#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    penguins_at[5].append(len(user_identified_penguins))
    print str(5) + "  -  " + str(len(user_identified_penguins))

    user_identified_penguins,clusters[20] = DivisiveDBSCAN(6).fit(user_markings[20],user_ips[20],debug=True)#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    penguins_at[20].append(len(user_identified_penguins))
    print str(20) + "  -  " + str(len(user_identified_penguins))

    notFound = 0
    total = 0
    for clusterIndex in range(len(clusters[5])):
        newCluster = []
        total += 1
        for pt in clusters[5][clusterIndex]:
            #find which cluster this pt wound up in after 20 classifications
            found = False

            for clusterIndex2 in range(len(clusters[20])):
                if pt in clusters[20][clusterIndex2]:
                    newCluster.append(clusterIndex2)
                    found = True
                    break
        if not(found):
            notFound += 1
        #print newCluster

    print "missing : " + str(notFound) + "," + str(total)