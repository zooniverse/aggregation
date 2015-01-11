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
import socket
import datetime
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

from divisiveKmeans import DivisiveKmeans
from agglomerativeClustering import agglomerativeClustering

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

client = pymongo.MongoClient()
db = client['condor_2014-11-23']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]
user_collection = db["condor_users"]

#create the list of subjects - so we can quickly find the indices of them
to_sample_from = list(subject_collection.find({"state":"complete"}))
subject_list = []
for zooniverse_id in random.sample(to_sample_from,100):
    bisect.insort(subject_list,zooniverse_id)

print "==--"
#create the list of users
all_users = []

for user_record in user_collection.find():
    if "name" in user_record:
        user = user_record["name"]
    else:
        user = user_record["ip"]

    bisect.insort(all_users,user)


print "****"


big_userList = []
big_subjectList = []
animal_count = 0

f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")

#subject_vote = {}
results = []
gold_values = {}


#to_sample_from2 = list(subject_collection.find({"classification_count":1,"state":"active"}))

print "===---"

votes = []

ip_index = 0
animal_count = -1

to_ibcc = []

real_animals = 0
fake_animals = 0

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

results_dict = {}

for subject_index,subject in enumerate(subject_list):
    print subject_index
    zooniverse_id = subject["zooniverse_id"]
    #print subject_index
    annotation_list = []
    user_list = []
    animal_list = []
    ip_addresses = []
    user_count = 0
    users_per_subject = []

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        user_count += 1
        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            if "user_name" in classification:
                user = classification["user_name"]
            else:
                user = classification["user_ip"]
                ip_addresses.append(user)

            #check to see if we have already encountered this subject/user pairing
            #due to some double classification errors
            if user in users_per_subject:
                continue
            users_per_subject.append(user)

            for animal in markings.values():
                scale = 1.875
                x = scale*float(animal["x"])
                y = scale*float(animal["y"])

                animal_type = animal["animal"]
                if not(animal_type in ["carcassOrScale","carcass","other",""]):
                    annotation_list.append((x,y))
                    #print annotation_list
                    user_list.append(user)
                    animal_list.append(animal_type)




        except (ValueError,KeyError):
            pass

    #print animal_list

    #if there were any markings on the image, use divisive kmeans to cluster the points so that each
    #cluster represents an image
    if annotation_list != []:
        # zooniverse_id = subject["zooniverse_id"]
        # url = subject["location"]["standard"]
        #
        # slash_index = url.rfind("/")
        # object_id = url[slash_index+1:]
        #
        # if not(os.path.isfile(base_directory+"/Databases/condors/images/"+object_id)):
        #     urllib.urlretrieve (url, base_directory+"/Databases/condors/images/"+object_id)
        #
        # image_file = cbook.get_sample_data(base_directory+"/Databases/condors/images/"+object_id)
        # image = plt.imread(image_file)
        #
        # fig, ax = plt.subplots()
        # im = ax.imshow(image)



        a = datetime.datetime.now()
        user_identified,clusters,users = DivisiveKmeans(1).fit2(annotation_list,user_list,debug=True)
        b = datetime.datetime.now()
        print "=="
        print len(user_identified)
        #for (x,y) in user_identified:
        #    plt.plot([x,],[y,],'.',color="red")
        c = datetime.datetime.now()
        user_identified = agglomerativeClustering(zip(annotation_list,user_list))
        d = datetime.datetime.now()
        print len(user_identified)
        print b-a
        print d-c
        print "--"
        #for (x,y) in user_identified:
        #    plt.plot([x-3,],[y-3,],'.',color="green")

        #plt.show()
        #plt.close()

