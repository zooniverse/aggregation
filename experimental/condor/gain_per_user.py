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
import random

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
db = client['condor_2014-11-06']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]


to_sample_from = list(subject_collection.find({"classification_count":{"$gte":8}}))

steps = [2,5,7,8]
condor_count =  {k:[] for k in steps}

for subject_count,subject in enumerate(random.sample(to_sample_from,10)):
    if subject_count == 10:
        break
    zooniverse_id = subject["zooniverse_id"]
    url = subject["location"]["standard"]

    slash_index = url.rfind("/")
    object_id = url[slash_index+1:]


    annotation_list = []

    user_markings = {k:[] for k in steps}
    user_list = {k:[] for k in steps}
    type_list = {k:[] for k in steps}

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):

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
                            if not(animal_type in ["carcassOrScale","carcass"]):
                                user_markings[s].append((x,y))
                                user_list[s].append(user_index)
                                type_list[s].append(animal_type)

                        except KeyError:
                            user_markings[s].append((x,y))
                            user_list[s].append(user_index)
                            type_list[s].append("NA")

        except ValueError:
            pass

    #do the divisive k means for each

    for s in steps:
        print s
        identified_animals,clusters = DivisiveKmeans(min(s,3)).fit2(user_markings[s],user_list[s],debug=True)

        #find out what kind of animal each person thought it was
        count = 0.
        for animal in clusters:
            type_vote = []
            condor_vote = 0
            for pt in animal:
                user_index = user_markings[s].index(pt)
                tt = type_list[s][user_index]
                #type_vote.append()
                if tt == "condor":
                    condor_vote += 1

            condor_percent = condor_vote/float(len(animal))
            if condor_percent >= 0.5:
                count += 1

        condor_count[s].append(count)


at_2 = [s/t for s,t in zip(condor_count[2],condor_count[8]) if t > 0]
at_5 = [s/t for s,t in zip(condor_count[5],condor_count[8]) if t > 0]
at_7 = [s/t for s,t in zip(condor_count[7],condor_count[8]) if t > 0]

mean_2 = np.mean(at_2)
mean_5 = np.mean(at_5)
mean_7 = np.mean(at_7)

plt.plot([2,5,7],[mean_2,mean_5,mean_7],'o-')
plt.show()




