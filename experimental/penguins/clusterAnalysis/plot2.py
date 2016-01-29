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

steps = [5,10,15,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
import random
#for subject in collection2.find({"classification_count": 20}):
for zooniverse_id in random.sample(to_sample,300):
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
    complete_annotations = []
    num_annotations = []
    for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
        user_index += 1
        if user_index == 21:
            break

        try:
            if classification["annotations"][2]["value"] == "complete":
                complete_annotations.append(True)
            else:
                complete_annotations.append(False)
        except KeyError:
            complete_annotations.append(False)

        per_user = []

        ip = classification["user_ip"]
        try:
            num_annotations.append(0)
            markings_list = classification["annotations"][1]["value"]
            if isinstance(markings_list,dict):
                for marking in markings_list.values():

                    if marking["value"] in ["adult","chick"]:
                        num_annotations[-1] += 1
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
    t =  zip(complete_annotations,num_annotations)
    print [c for n,c in t if n]
    print [c for n,c in t if not(n)]
    if user_markings[5] == []:
        print "skipping empty"
        subject_index += -1
        continue

    for s in steps:

        user_identified_penguins = DivisiveDBSCAN(3).fit(user_markings[s],user_ips[s])#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")
        penguins_at[s].append(len(user_identified_penguins))
        print str(s) + "  -  " + str(len(user_identified_penguins))

    # url = subject["location"]["standard"]
    # object_id= str(subject["_id"])
    # image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"
    # if not(os.path.isfile(image_path)):
    #     urllib.urlretrieve(url, image_path)
    #
    # image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    # image = plt.imread(image_file)
    # fig, ax = plt.subplots()
    # im = ax.imshow(image)
    # plt.show()

    if subject_index == 30:
        break


pickle.dump(penguins_at,open(base_directory+"/Databases/penguins_at.pickle","wb"))

# max5_10 = {}
# for x,y in zip(penguins_at[5],penguins_at[10]):
#     if not(x in max5_10):
#         max5_10[x] = y
#     else:
#         max5_10[x] = max(max5_10[x],y)
#
# print max5_10
#
# max10_15 = {}
# for x,y in zip(penguins_at[10],penguins_at[15]):
#     if not(x in max5_10):
#         max5_10[x] = y
#     else:
#         max5_10[x] = max(max5_10[x],y)



#fig, (ax0, ax1) = plt.subplots(nrows=2)
#plt.plot(penguins_at[5],penguins_at[10],'.')
#plt.plot(penguins_at[10],penguins_at[15],'.',color="green")
#plt.plot((0,100),(0,100))
#plt.show()