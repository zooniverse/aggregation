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
db = client['condor_2014-11-11']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]


to_sample_from = list(subject_collection.find({"tutorial":{"$ne":True},"state":"complete","metadata.retire_reason":{"$nin":["blank","no_condors_present","blank_consensus"]}}))

steps = [2,5,7,8]
condor_count =  {k:[] for k in steps}

l1 = []
l2 = []

for subject_count,subject in enumerate(random.sample(to_sample_from,500)):
    zooniverse_id = "ACW0004yfc"
    zooniverse_id = subject["zooniverse_id"]
    url = subject["location"]["standard"]
    slash_index = url.rfind("/")
    object_id = url[slash_index+1:]

    annotation_list = []
    user_list = []
    animals = []

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


                try:
                    animal_type = animal["animal"]
                    animals.append(animal_type)
                    #if not(animal_type in ["carcassOrScale","carcass"]):
                    if animal_type in ["condor","turkeyVulture","goldenEagle"]:
                        annotation_list.append((x,y))

                        user_list.append(user)


                except KeyError:
                    pass

        except ValueError:
            pass

    if annotation_list == []:
        continue

    if not(os.path.isfile(base_directory+"/Databases/condors/images/"+object_id)):
        urllib.urlretrieve (url, base_directory+"/Databases/condors/images/"+object_id)
    user_identified_condors,clusters = DivisiveKmeans(3).fit2(annotation_list,user_list,debug=True)
    f_name = base_directory+"/Databases/condors/images/"+object_id
    if user_identified_condors == []:
        pass
    else:
        l2.append(subject["classification_count"])
        l1.append(subject["zooniverse_id"])
        #print animals
        print l1
        print l2
        DivisiveKmeans(3).__fix__(user_identified_condors,clusters,annotation_list,user_list,200,f_name)



