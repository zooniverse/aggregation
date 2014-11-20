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
import math

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-11']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

relations = []
one = []

#print subject_collection.count({"classification_count":{"$gt":1}})
sites = {}

for subject in subject_collection.find({"state":"complete"}):
    f = subject["metadata"]["file"]
    zooniverse_id = subject["zooniverse_id"]

    slash = f.rfind("/")
    site = f[:slash]
    if not(site in sites):
        sites[site] = [zooniverse_id]
    else:
        sites[site].append(zooniverse_id)


for s in sites:
    for zooniverse_id in s:
        annotation_list = []
        user_list = []
        for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
            try:
                mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
                markings = classification["annotations"][mark_index].values()[0]

                for animal in markings.values():
                    scale = 1.875
                    x = scale*float(animal["x"])
                    y = scale*float(animal["y"])
                    try:
                        animal_type = animal["animal"]
                        #if not(animal_type in ["carcassOrScale","carcass"]):
                        if animal_type == "condor":
                            annotation_list.append((x,y))
                            user_list.append(user_index)

                    except KeyError:
                        annotation_list.append((x,y))
                        user_list.append(user_index)

            except ValueError:
                pass

        user_identified_condors,clusters = DivisiveKmeans(3).fit2(annotation_list,user_list,debug=True)

        relations = []
        if len(user_identified_condors) > 1:
            for c1_index in range(len(clusters)):
                for c2_index in range(c1_index+1,len(clusters)):


                    condor1 = user_identified_condors[c1_index]
                    condor2 = user_identified_condors[c2_index]

                    dist = math.sqrt((condor1[0]-condor2[0])**2+(condor1[1]-condor2[1])**2)
                    users_1 = [user_list[annotation_list.index(pt)] for pt in clusters[c1_index]]
                    users_2 = [user_list[annotation_list.index(pt)] for pt in clusters[c2_index]]

                    overlap = [u for u in users_1 if u in users_2]
                    if len(overlap) <= 1:
                        relations.append((dist,len(overlap),c1_index,c2_index))