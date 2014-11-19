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

num_condors = []

with open(base_directory+"/Documents/condor_gold","r") as f:
    while True:
        zooniverse_id = f.readline()
        condors = f.readline()

        if not zooniverse_id:
            break

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

        user_identified_condors = DivisiveKmeans(3).fit2(annotation_list,user_list)#,jpeg_file=base_directory+"/Databases/condors/images/"+object_id)
        if len(user_identified_condors) > 0:
            num_condors.append(len(user_identified_condors))

#n, bins, patches = plt.hist(num_condors, 10, facecolor='green')
incorrect_num_condors = []
with open(base_directory+"/Documents/condor_error","r") as f:
    while True:
        zooniverse_id = f.readline()
        condors = f.readline()

        if not zooniverse_id:
            break

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

        user_identified_condors = DivisiveKmeans(3).fit2(annotation_list,user_list)#,jpeg_file=base_directory+"/Databases/condors/images/"+object_id)
        if len(user_identified_condors) > 0:
            incorrect_num_condors.append(len(user_identified_condors))
tt = zip(num_condors,incorrect_num_condors)
print tt
print np.array(tt)
n_2, bins, patches = plt.hist([num_condors,incorrect_num_condors], 10, histtype='bar')
#n_2, bins, patches =plt.hist(incorrect_num_condors, bins, facecolor='red')
#print n_2
plt.close()

# print n_2[0]
# print n_2[1]
# center = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
# Y = [a/float(a+b) for a,b, in zip(n_2[0],n_2[1])]
# print center
# print Y
# plt.plot(center,Y,"o-")
# plt.show()
#
upper_bound = [x+1 for x in incorrect_num_condors]
# Y = num_condors[:]
# Y.extend(upper_bound)
# n, bins, patches = plt.hist(Y, 10, normed=1,histtype='step', cumulative=True)
lower_bound = [x-1 for x in incorrect_num_condors]
# Y = num_condors[:]
# Y.extend(lower_bound)
# n, bins, patches = plt.hist(Y, 10, normed=1,histtype='step', cumulative=True)
# plt.show()

actually_counted = num_condors[:]
actually_counted.extend(incorrect_num_condors)

true_values = num_condors[:]
true_values.extend(upper_bound)

percentage = [a/float(t) for a,t in zip(actually_counted,true_values)]
n, bins, patches = plt.hist(percentage, 10, normed=1,histtype='step', cumulative=True)

true_values = num_condors[:]
true_values.extend(lower_bound)
percentage = [a/float(max(t,1)) for a,t in zip(actually_counted,true_values)]
n, bins, patches = plt.hist(percentage, 10, normed=1,histtype='step', cumulative=True)

plt.show()
