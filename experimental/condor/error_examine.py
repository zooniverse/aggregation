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

with open(base_directory+"/Dropbox/condor_error","r") as f:
    while True:
        zooniverse_id = f.readline()
        condors = f.readline()

        if not zooniverse_id:
            break

        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id[:-1]})
        print subject["metadata"]["file"]


        url = subject["location"]["standard"]

        slash_index = url.rfind("/")
        object_id = url[slash_index+1:]



        continue

        annotation_list = []
        user_list = []

        for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id[:-1]})):

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

        user_identified_condors,clusters = DivisiveKmeans(3).fit2(annotation_list,user_list,debug=True)#,jpeg_file=base_directory+"/Databases/condors/images/"+object_id)

        image_file = cbook.get_sample_data(base_directory+"/Databases/condors/images/"+object_id)
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)
        #plt.show()

        print len(clusters)
        relations = []
        for c1_index in range(len(clusters)):
            for c2_index in range(c1_index+1,len(clusters)):


                condor1 = user_identified_condors[c1_index]
                condor2 = user_identified_condors[c2_index]

                dist = math.sqrt((condor1[0]-condor2[0])**2+(condor1[1]-condor2[1])**2)
                users_1 = [user_list[annotation_list.index(pt)] for pt in clusters[c1_index]]
                users_2 = [user_list[annotation_list.index(pt)] for pt in clusters[c2_index]]

                overlap = [u for u in users_1 if u in users_2]
                relations.append((dist,len(overlap),c1_index,c2_index))

        relations.sort(key= lambda x:x[0])

        if user_identified_condors != []:
            x,y = zip(*user_identified_condors)
            plt.plot(x,y,'.',color='blue')

            print relations[:10]
            for i in range(min(len(relations),10)):
                pt1 = user_identified_condors[relations[i][2]]
                pt2 = user_identified_condors[relations[i][3]]

                if relations[i][1] <= 1:
                    plt.plot((pt1[0],pt2[0]),(pt1[1],pt2[1]),color="red")

            plt.show()
        else:
            plt.close()