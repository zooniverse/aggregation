#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import cPickle as pickle
import bisect
import csv
import matplotlib.pyplot as plt
import random
import math
import urllib
import matplotlib.cbook as cbook


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/classifier")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
    sys.path.append("/home/greg/github/reduction/experimental/classifier")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from divisiveKmeans import DivisiveKmeans
from iterativeEM import IterativeEM

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-23']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]




big_userList = []
big_subjectList = []
animal_count = 0

f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")
alreadyDone = []
animals_in_image = {}
animal_index = -1
global_user_list = []
animal_to_image = []
zooniverse_list = []
condor_votes = {}
animal_votes = {}
#subject_vote = {}
results = []

to_sample_from = list(subject_collection.find({"state":"complete"}))
to_sample_from2 = list(subject_collection.find({"classification_count":1,"state":"active"}))

votes = []

sample = random.sample(to_sample_from,100)
#sample.extend(random.sample(to_sample_from2,1000))
# for subject_index,subject in enumerate(sample):
#     print "== " + str(subject_index)
#     zooniverse_id = subject["zooniverse_id"]
#     for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
#         if "user_name" in classification:
#             user = classification["user_name"]
#         else:
#             user = classification["user_ip"]
#
#         try:
#             tt = index(big_userList,user)
#         except ValueError:
#             bisect.insort(big_userList,user)




for subject_index,subject in enumerate(sample):
    print subject_index
    zooniverse_id = subject["zooniverse_id"]

    annotation_list = []
    user_list = []
    animal_list = []
    #local_users = []

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            if "user_name" in classification:
                user = classification["user_name"]
            else:
                user = classification["user_ip"]
            found_condor = False

            for animal in markings.values():
                scale = 1.875
                x = scale*float(animal["x"])
                y = scale*float(animal["y"])

                animal_type = animal["animal"]
                if not(animal_type in ["carcassOrScale","carcass"]):
                    annotation_list.append((x,y))
                    #print annotation_list
                    user_list.append(user)
                    animal_list.append(animal_type)
                    if not(user in global_user_list):
                        global_user_list.append(user)
                    #local_users.append(user)

                if animal_type == "condor":
                    found_condor = True


        except (ValueError,KeyError):
            pass

    #if there were any markings on the image, use divisive kmeans to cluster the points so that each
    #cluster represents an image
    if annotation_list != []:
        user_identified,clusters = DivisiveKmeans(3).fit2(annotation_list,user_list,debug=True)

        #fix split clusters if necessary
        if user_identified != []:
            user_identified,clusters = DivisiveKmeans(3).__fix__(user_identified,clusters,annotation_list,user_list,200)

            for center,c in zip(user_identified,clusters):
                animal_index += 1
                #animal_votes.append([])
                animal_to_image.append(zooniverse_id)

                if not(zooniverse_id in animals_in_image):
                    animals_in_image[zooniverse_id] = [animal_index]
                else:
                    animals_in_image[zooniverse_id].append(animal_index)

                results.append((zooniverse_id,center))
                for pt in c:
                    pt_index = annotation_list.index(pt)
                    user_index = global_user_list.index(user_list[pt_index])
                    animal_type = animal_list[annotation_list.index(pt)]

                    if animal_type == "condor":
                        votes.append((user_index,animal_index,1))
                        if not(animal_index in animal_votes):
                            animal_votes[animal_index] = [1]
                        else:
                            animal_votes[animal_index].append(1)
                    else:
                        votes.append((user_index,animal_index,0))
                        if not(animal_index in animal_votes):
                            animal_votes[animal_index] = [0]
                        else:
                            animal_votes[animal_index].append(0)


print "=====---"
#print votes
classify = IterativeEM()
classify.__classify__(votes)

most_likely = classify.__getMostLikely__()
estimates = classify.__getEstimates__()

X = []
Y = []
X2 = []
Y2 = []

#for subject_index,zooniverse_id in enumerate(big_subjectList):
for ii in range(animal_index):
    x = np.mean(animal_votes[ii])

    y = estimates[ii][1]
    X.append(x)
    Y.append(y)

    if math.fabs(x-y) > 0.3:
        zooniverse_id,(centerX,centerY) = results[ii]
        print x,y
        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})

        url = subject["location"]["standard"]
        slash_index = url.rfind("/")
        object_id = url[slash_index+1:]

        if not(os.path.isfile(base_directory+"/Databases/condors/images/"+object_id)):
            urllib.urlretrieve (url, base_directory+"/Databases/condors/images/"+object_id)

        image_file = cbook.get_sample_data(base_directory+"/Databases/condors/images/"+object_id)
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)
        plt.plot([centerX,],[centerY,],'o')
        plt.show()

    # #if ((x < 0.5) and (y > 0.5)) or ((x > 0.5) and (y < 0.5)):
    #     subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    #     print x,y
    #     print subject["location"]["standard"]

    #    #print most_likely[subject_index],estimates[subject_index],np.mean(subject_vote[zooniverse_id])
    #else:
    #    print estimates[subject_index],0

plt.plot(X,Y,'.',color="blue")
plt.plot(X2,Y2,'.',color="red")
plt.xlim((-0.05,1.05))
plt.ylim((-0.05,1.05))
plt.show()