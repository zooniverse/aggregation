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

#print subject_collection.count({"classification_count":{"$gt":1}})

#for subject in subject_collection.find({"classification_count":{"$gt":1}}):
#    #if not("USFWS photos/Remote Feeding Site Photos/Remote Feeding Photos_2008/Bitter Creek/NRFS/NRFS 4.16-4.17.2008=CORA, 17CACO/" in subject["metadata"]["file"]):
#    if not("Pinnacles photos/PINN Photographs/Command Post Camera/2010/CommPost20101129_20101130/" in subject["metadata"]["file"]):
#        continue
for zooniverse_id in [u'ACW00013g1', u'ACW00013es', u'ACW00013fe', u'ACW00013fs', u'ACW00013ci', u'ACW00013bk', u'ACW00013ea', u'ACW00013by', u'ACW00013d6', u'ACW00013co', u'ACW00013eq', u'ACW00013de', u'ACW00013bw', u'ACW00013e8', u'ACW00013ch', u'ACW00013cu', u'ACW00013cv', u'ACW00013c6', u'ACW00013bl', u'ACW00013d0', u'ACW00013dt', u'ACW000137p', u'ACW000136k', u'ACW0001373', u'ACW0001361', u'ACW0001376', u'ACW000138v', u'ACW000137f', u'ACW000137y', u'ACW000136j', u'ACW000137m', u'ACW000138l', u'ACW0001356', u'ACW000138e', u'ACW000133v', u'ACW000136e']:
    zooniverse_id = "ACW00011h2"
    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    #zooniverse_id = subject["zooniverse_id"]

    print zooniverse_id
    print subject["classification_count"]
    url = subject["location"]["standard"]
    slash_index = url.rfind("/")
    object_id = url[slash_index+1:]

    annotation_list = []
    user_list = []

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if user == "wreness":
            print classification

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

                        user_list.append(user)

                except KeyError:
                    annotation_list.append((x,y))
                    user_list.append(user)

        except ValueError:
            pass

    #user_identified_condors,clusters = DivisiveKmeans(3).fit2(annotation_list,user_list,debug=True)
    relations = []

    if not(os.path.isfile(base_directory+"/Databases/condors/images/"+object_id)):
        urllib.urlretrieve (url, base_directory+"/Databases/condors/images/"+object_id)

    user_identified_condors = DivisiveKmeans(1).fit2(annotation_list,user_list,jpeg_file=base_directory+"/Databases/condors/images/"+object_id)

    image_file = cbook.get_sample_data(base_directory+"/Databases/condors/images/"+object_id)
    print object_id
    image = plt.imread(image_file)

    fig, ax = plt.subplots()
    im = ax.imshow(image)

    x,y = zip(*annotation_list)
    plt.plot(x,y,'.',color='yellow')
    x,y = zip(*user_identified_condors)
    plt.plot(x,y,'.',color='blue')

    #DivisiveKmeans(3).__fix__(user_identified_condors,clusters,annotation_list,user_list,200,base_directory+"/Databases/condors/images/"+object_id)
    #continue
    plt.show()
    break

    if len(user_identified_condors) > 1:
        for c1_index in range(len(clusters)):
            for c2_index in range(c1_index+1,len(clusters)):


                condor1 = user_identified_condors[c1_index]
                condor2 = user_identified_condors[c2_index]

                dist = math.sqrt((condor1[0]-condor2[0])**2+(condor1[1]-condor2[1])**2)
                users_1 = [user_list[annotation_list.index(pt)] for pt in clusters[c1_index]]
                users_2 = [user_list[annotation_list.index(pt)] for pt in clusters[c2_index]]

                overlap = [u for u in users_1 if u in users_2]
                #print (len(overlap),dist)

                if (len(overlap) <= 1):
                    #print (len(overlap),dist)
                    if (dist <= 60):
                    #relations.append((dist,len(overlap),c1_index,c2_index))
                        if len(overlap) == 1:
                            print overlap
                            plt.plot((condor1[0],condor2[0]),(condor1[1],condor2[1]),'o-',color="red")
                        else:
                            plt.plot((condor1[0],condor2[0]),(condor1[1],condor2[1]),'o-',color="salmon")
                    else:
                        if len(overlap) == 1:
                            pass
                            #plt.plot((condor1[0],condor2[0]),(condor1[1],condor2[1]),'o-',color="green")
                        else:
                            plt.plot((condor1[0],condor2[0]),(condor1[1],condor2[1]),'o-',color="lime")

    plt.show()

