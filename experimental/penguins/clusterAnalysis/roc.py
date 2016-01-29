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
from clusterCompare import cluster_compare

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
noise_list = {k:[] for k in steps}
overall_found= {s:[[],[],[],[],[]] for s in [5,10,15]}
overall_fake = {s:[[],[],[],[],[]] for s in [5,10,15]}

with open("/home/greg/Documents/gold_standard_penguins","rb") as gold_standard:
    for line_index, line in enumerate(gold_standard.readlines()):
        if line_index == 30:#41:
            break

        zooniverse_id, num_markings = line.split(" ")
        num_markings = int(num_markings[:-1])
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


        penguins = []
        penguins_center = {}
        noise_points = {}
        #gold standard
        gold_centers,gold_clusters,noise__ = DivisiveDBSCAN(num_markings).fit(user_markings[20],user_ips[20],debug=True)#,jpeg_file=base_directory + "/Databases/penguins/images/"+object_id+".JPG")
        print "gold standard number " + str(len(gold_clusters))


        #not_found = []
        fake_penguins = {}
        for s in [5,10,15]:
            print "== " + str(s)
            #not_found.append([])
            fake_penguins[s] = []

            for nn in [1,2,3,4,5]:
                user_identified_penguins,penguin_clusters,noise__ = DivisiveDBSCAN(nn).fit(user_markings[s],user_ips[s],debug=True)

                #missed penguins - in gold standard but not found
                not_found = cluster_compare(penguin_clusters,gold_clusters)

                #fake penguins - found but not actually real
                fake_found = cluster_compare(gold_clusters,penguin_clusters)

                print len(not_found),len(fake_found)
                #not_found[-1].append(len(gold_clusters) -len(not_found))
                fake_penguins[s].append(len(fake_found))

                overall_found[s][nn-1].append((len(gold_clusters) - len(not_found))/float(len(gold_clusters)))

        #false positives - fake penguins - can only be calculated after everything else is done
        for s in [5,10,15]:
            for nn in [1,2,3,4,5]:
                if fake_penguins[15][0] == 0:
                    overall_fake[s][nn-1].append(0)
                else:
                    overall_fake[s][nn-1].append(fake_penguins[s][nn-1]/float(fake_penguins[15][0]))


for s in [5,10,15]:
    print overall_fake[s]
    print overall_found[s]
    #X = false positive rates
    X = [np.mean(overall_fake[s][nn-1]) for nn in [1,2,3,4,5]]
    #Y = true positive rates
    Y = [np.mean(overall_found[s][nn-1]) for nn in [1,2,3,4,5]]
    print X
    print Y
    plt.plot(X,Y,'-o')

plt.show()