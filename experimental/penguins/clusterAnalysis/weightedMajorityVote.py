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
from copy import deepcopy

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
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
#to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
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
file_out = "/Databases/penguins_vote_.pickle"
f = open("/home/greg/Documents/new_gold","rb")

completed_subjects = []

for subject in collection2.find({"classification_count":20}):
    zooniverse_id = subject["zooniverse_id"]
    if subject["metadata"]["counters"]["animals_present"] > 10:
        completed_subjects.append(zooniverse_id)

for subject_index,zooniverse_id in enumerate(random.sample(completed_subjects,200)):
    if subject_index == 5:
        break

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
        #print ip
        try:
            markings_list = classification["annotations"][1]["value"]
            if isinstance(markings_list,dict):
                for marking in markings_list.values():
                    #print marking
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

    penguins = []
    penguins_center = {}
    noise_points = {}
    #gold_penguins,gold_clusters,noise__ = DivisiveDBSCAN(6).fit(user_markings[s],user_ips[s],debug=True)

    #if len(gold_clusters) == 0:
    #    continue
    #gold_penguin_count.append(len(gold_clusters))

    #print "gold  -  " + str(len(gold_penguins))
    for s in steps:
        if s == 20:
            user_identified_penguins,penguin_clusters,noise__ = DivisiveDBSCAN(1).fit(user_markings[s],user_ips[s],debug=True)
        else:
            user_identified_penguins,penguin_clusters,noise__ = DivisiveDBSCAN(1).fit(user_markings[s],user_ips[s],debug=True)



        penguins_at[s].append(deepcopy(penguin_clusters))
        #penguins_center[s] = user_identified_penguins
        #noise_list[s].append(noise)

        #penguins.append(penguin_clusters)
        #print penguin_clusters
        #print noise__
        #noise_points[s] = [x for x,u in noise__]
        print str(s) + "  -  " + str(len(user_identified_penguins))
        #if len(user_identified_penguins) > 20:
        #    break


    #if len(user_identified_penguins) == 0:
    #    continue

    # if len(user_identified_penguins) <= 80:
    #     #print noise__
    #     #not_found = cluster_compare(penguins[0],penguins[-1])
    #     #if not_found == []:
    #     #    continue
    #
    #
    #
    #     image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    #     image = plt.imread(image_file)
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(image)
    #
    #     # try:
    #     #     X,Y = zip(*penguins_center[5])
    #     #     plt.plot(X,Y,'.',color="red")
    #     # except ValueError:
    #     #     pass
    #     #
    #     # X,Y = zip(*noise_points[5])
    #     # plt.plot(X,Y,'.',color="green")
    #     # print [(x,y) for i,(x,y) in enumerate(user_identified_penguins) if i in not_found]
    #     # X,Y = zip(*[(x,y) for i,(x,y) in enumerate(user_identified_penguins) if i in not_found])
    #     # #X,Y = zip(*noise)
    #     #
    #     # plt.plot(X,Y,'.',color="blue")
    #     plt.show()


    if (subject_index % 5) == 0:
        print "WRITING"
        pickle.dump(penguins_at,open(base_directory+file_out,"wb"))

pickle.dump(penguins_at,open(base_directory+file_out,"wb"))

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