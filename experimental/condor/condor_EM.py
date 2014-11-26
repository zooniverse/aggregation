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
animal_votes = []
subject_vote = {}

to_sample_from = list(subject_collection.find({"state":"complete"}))
to_sample_from2 = list(subject_collection.find({"classification_count":1,"state":"active"}))

votes = []

sample = random.sample(to_sample_from,1500)
sample.extend(random.sample(to_sample_from2,1500))
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
    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if not(user in big_userList):
            big_userList.append(user)
        if user_index == 0:
        #if not(zooniverse_id in big_subjectList):
            big_subjectList.append(zooniverse_id)

        user_index = big_userList.index(user)
        subject_index = len(big_subjectList)-1 #.index(zooniverse_id)

        try:
            tt = index(alreadyDone,(user_index,subject_index))
            continue
        except ValueError:
            bisect.insort(alreadyDone,(user_index,subject_index))


        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            found = False
            for animal in markings.values():


                try:
                    animal_type = animal["animal"]
                    if animal_type in ["condor"]:
                        found = True
                        break

                except KeyError:
                    pass


            if found:
                votes.append((user_index,subject_index,1))
                if not(zooniverse_id in subject_vote):
                    subject_vote[zooniverse_id] = [1]
                else:
                    subject_vote[zooniverse_id].append(1)
            else:
                votes.append((user_index,subject_index,0))
                if not(zooniverse_id in subject_vote):
                    subject_vote[zooniverse_id] = [0]
                else:
                    subject_vote[zooniverse_id].append(0)

        except ValueError:
            votes.append((user_index,subject_index,0))
            if not(zooniverse_id in subject_vote):
                subject_vote[zooniverse_id] = [0]
            else:
                subject_vote[zooniverse_id].append(0)

print "=====---"
classify = IterativeEM()
classify.__classify__(votes)

most_likely = classify.__getMostLikely__()
estimates = classify.__getEstimates__()

X = []
Y = []
X2 = []
Y2 = []

for subject_index,zooniverse_id in enumerate(big_subjectList):
    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    if zooniverse_id in subject_vote:
        x = np.mean(subject_vote[zooniverse_id])

        y = estimates[subject_index][1]
        if subject["state"] == "complete":
            X.append(x)
            Y.append(y)
        else:
            X2.append(x)
            Y2.append(y)
            if math.fabs(x-y) > 0.3:
            #if ((x < 0.5) and (y > 0.25)) or ((x > 0.5) and (y < 0.75)):
                print x,y
                print subject["location"]["standard"]

    # if math.fabs(x-y) > 0.3:
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