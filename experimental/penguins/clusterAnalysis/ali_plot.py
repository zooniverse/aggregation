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
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from clusterCompare import metric,metric2

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]


to_sample = list(subject_collection.find({"classification_count":20}))

steps = [5,10,15,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import random
#for subject in collection2.find({"classification_count": 20}):
noise_list = {k:[] for k in steps}
gold_penguin_count = []
#gold_standard = open("/home/greg/Documents/gold_standard_penguins","rb")
#for line_index, line in enumerate(gold_standard.readlines()):
#    if line_index == 40:#41:
#        break
#
#    zooniverse_id, num_markings = line.split(" ")
#    num_markings = int(num_markings[:-1])

for subject_index,subject in enumerate(random.sample(to_sample,20)):

    #zooniverse_id = subject["zooniverse_id"]
    print "=== " + str(subject_index)
    zooniverse_id = subject["zooniverse_id"]

    alreadyThere = True
    user_markings = {k:[] for k in steps}
    user_ips = {k:[] for k in steps}

    user_index = 0
    for classification in classification_collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
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

    #if user_markings[5] == []:
    #    print "skipping empty"
    #    subject_index += -1
    #    continue

    # url = subject["location"]["standard"]
    # object_id= str(subject["_id"])
    # image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"
    # if not(os.path.isfile(image_path)):
    #     urllib.urlretrieve(url, image_path)

    if user_markings[20] == []:
        continue


    penguins = []
    penguins_center = {}
    noise_points = {}
    gold_penguins,gold_clusters,noise__ = DivisiveDBSCAN(3).fit(user_markings[s],user_ips[s],debug=True)

    if len(gold_penguins) == 0:
        continue

    if len(gold_penguins) > 50:
        continue

    #if len(gold_clusters) == 0:
    #    continue
    gold_penguin_count.append(len(gold_clusters))


    for s in [5,10,15]:
        print s
        user_identified_penguins,penguin_clusters,noise__ = DivisiveDBSCAN(3).fit(user_markings[s],user_ips[s],debug=True)



        penguins_at[s].append(len(user_identified_penguins)/float(len(gold_penguins)))

mean_values = [np.mean(penguins_at[s]) for s in steps[:-1]]
standard_error = [np.std(penguins_at[s])/float(len(penguins_at[s])) for s in steps[:-1]]
plt.errorbar(steps[:-1],mean_values,yerr=standard_error,fmt="-o")
plt.show()