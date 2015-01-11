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

subject = subject_collection.find_one({"state":"active"})
zooniverse_id = subject["zooniverse_id"]



for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}).limit(1):
    main_user = classification["user_name"]
    print main_user

    to_sample = set([])
    for classification2 in classification_collection.find({"user_name":main_user,"tutorial":False}).limit(40000):
        zooniverse_id = classification2["subjects"][0]["zooniverse_id"]
        #check to see if this image has been completed
        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
        if subject["state"] == "complete":
            if not("blank" in subject["metadata"]["counters"]) or (subject["metadata"]["counters"]["blank"] != subject["classification_count"]):
                to_sample.add(zooniverse_id)



    print len(list(to_sample))

    for zooniverse_id in list(to_sample)[0:10]:
        print "///////////----"
        print zooniverse_id
        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
        #print subject["metadata"]["counters"],subject["classification_count"]

        subject_results = []

        for classification2 in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
            #if "user_name" in classification2:
            #    print classification2["user_name"],not("finished_at" in classification2["annotations"][0])
            #print classification2["annotations"][0]

            for marking in classification2["annotations"]:
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

                if "user_name" in classification2:
                    user = classification2["user_name"]
                else:
                    user = classification2["user_ip"]

                subject_results.append((pt,species,user))

        if subject_results != []:
            #print subject_results
            pts,species,users = zip(*subject_results)

            plankton,clusters,users_l = DivisiveKmeans(1).fit2(pts,users,debug=True)
            plankton,clusters = DivisiveKmeans(1).__fix__(plankton,clusters,pts,users,0)

            print users
            empty = True
            for c in clusters:
                if len(c) >= 3:
                    empty = False
                for m in c:
                    index = pts.index(m)
                    print species[index],users[index]

                print "--"

            if not(main_user in users) and not(empty):
                print classification_collection.find_one({"subjects.zooniverse_id":zooniverse_id,"user_name":main_user})
                assert False