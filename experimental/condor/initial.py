#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import random

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"


sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-20']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

to_sample_from = list(subject_collection.find({"tutorial":{"$ne":True},"state":"complete","metadata.retire_reason":{"$nin":["blank","no_condors_present","blank_consensus"]}}))

steps = [3,20]
error_count = 0
total = 0
for subject in random.sample(to_sample_from,200):
    zooniverse_id = subject["zooniverse_id"]

    user_markings = {k:[] for k in steps}
    user_list = {k:[] for k in steps}
    type_list = {k:[] for k in steps}
    animals_per_user = {k:[] for k in steps}
    max_animals = {k:{} for k in steps}


    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if user in user_list:
            continue

        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            for animal in markings.values():
                scale = 1.875
                x = scale*float(animal["x"])
                y = scale*float(animal["y"])

                for s in steps:
                    if user_index < s:
                        #only add the animal if it is not a
                        try:
                            animal_type = animal["animal"]
                            if animal_type in ["condor","turkeyVulture","goldenEagle"]:
                                user_markings[s].append((x,y))
                                user_list[s].append(user_index)
                                type_list[s].append(animal_type)

                                if user in max_animals[s]:
                                    max_animals[s][user] += 1
                                else:
                                    max_animals[s][user] = 1


                        except KeyError:
                            pass


        except ValueError:
            pass

    #gold standard
    if user_markings[20] != []:
        gold,gold_clusters = DivisiveKmeans(3).fit2(user_markings[20],user_list[20],debug=True)

        if gold != []:
            gold,gold_clusters = DivisiveKmeans(3).__fix__(gold,gold_clusters,user_markings[20],user_list[20],200)
    else:
        gold = []
        gold_clusters = []

    first_step = steps[0]

    if user_markings[first_step] != []:
        identified_animals,clusters = DivisiveKmeans(first_step).fit2(user_markings[first_step],user_list[first_step],debug=True)

        if identified_animals != []:
            identified_animals,clusters = DivisiveKmeans(2).__fix__(identified_animals,clusters,user_markings[first_step],user_list[first_step],200)
    else:
        identified_animals = []

    if (len(gold) > 0) and (len(identified_animals) == 0):

        print subject["location"]["standard"]
        if max_animals[first_step].values() == []:
            print 0
            error_count += 1
        else:
            print max(max_animals[first_step].values())

        # for c in gold_clusters:
        #     print "===//"
        #     print len(c)
        #     for pt in c:
        #         ind = user_markings[20].index(pt)
        #         print type_list[20][ind]

    if len(gold) > 0:
        total += 1
    #print len(identified_animals),len(gold)
    #print len(identified_animals)/float(max(max_animals[2].values()))
print error_count
print total




