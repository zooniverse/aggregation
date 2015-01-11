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

def dist(x1,x2):
    return math.sqrt((x1[0]-x2[0])**2+(x1[0]-x2[0])**2)

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

#load in the gold standard data
current_zooniverse_id = None
goldStandard = {}
with open("/home/greg/Documents/condor_gold","rb") as f:
    for l in f.readlines():
        if not(" " in l):
            current_zooniverse_id = l[:-1]
            goldStandard[current_zooniverse_id] = [[],[]]
        else:
            assert current_zooniverse_id is not None
            vals = l[:-1].split(' ')
            goldStandard[current_zooniverse_id][0].append((float(vals[0]),float(vals[1])))
            goldStandard[current_zooniverse_id][1].append(vals[2]=="1")

all_users = []

for user_record in user_collection.find():
    if "name" in user_record:
        user = user_record["name"]
    else:
        user = user_record["ip"]

    bisect.insort(all_users,user)


# for classification in classification_collection.find():
#     if classification["subjects"] == []:
#         continue
#
#     if "user_name" in classification:
#         user = classification["user_name"]
#     else:
#         user = classification["user_ip"]
#
#     zooniverse_id = classification["subjects"][0]["zooniverse_id"]
#
#     #check to see if this classification corresponds to one of the subjects we are sampling
#     try:
#         index(subject_list,zooniverse_id)
#
#         #have we already encountered this subject?
#         try:
#             index(user_list,user)
#         except ValueError:
#             bisect.insort(user_list,user)
#     except ValueError:
#         continue


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

for subject_index,zooniverse_id in enumerate(goldStandard.keys()):
    print "-" + str(subject_index)
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
        if user_count == 2:
            break
    print user_count
    #print animal_list

    #if there were any markings on the image, use divisive kmeans to cluster the points so that each
    #cluster represents an image
    if annotation_list != []:
        user_identified,clusters,users = DivisiveKmeans(1).fit2(annotation_list,user_list,debug=True)

        #fix split clusters if necessary
        if user_identified != []:
            user_identified,clusters,users_per_cluster = DivisiveKmeans(3).__fix__(user_identified,clusters,users,200)
            pos = 0
            neg = 0

            results_dict[zooniverse_id] = []

            #find out which users marked this "animal"
            for c,users_l in zip(user_identified,users_per_cluster):
                #moving on to the next animal so increase counter
                animal_count += 1
                results_dict[zooniverse_id].append((c,animal_count,users_l,user_count))

                for u in users_per_subject:
                    try:
                        socket.inet_aton(u)
                        i = ip_addresses.index(u) + ip_index

                        if u in users_l:
                            to_ibcc.append((-i,animal_count,1))
                            pos += 1
                        else:
                            to_ibcc.append((-i,animal_count,0))
                            neg += 1
                    except (socket.error,UnicodeEncodeError) as e:

                        if u in users_l:
                            to_ibcc.append((index(all_users,u),animal_count,1))
                            pos += 1
                        else:
                            to_ibcc.append((index(all_users,u),animal_count,0))
                            neg += 1

                if pos > neg:
                    real_animals += 1

                    true_pos += pos/float(pos+neg)
                    false_neg += neg/float(pos+neg)
                else:
                    fake_animals += 1

                    false_pos += pos/float(pos+neg)
                    true_neg += neg/float(pos+neg)

    ip_index += len(ip_addresses)
    if subject_index == 200:
        break

ibcc_user_list = []
for u,animal_index,found in to_ibcc:
    if not(u in ibcc_user_list):
        ibcc_user_list.append(u)
    # try:
    #     i = index(ibcc_user_list,u)
    # except ValueError:
    #     bisect.insort(ibcc_user_list,u)

for u,animal_index,found in to_ibcc:
    i = ibcc_user_list.index(u)
    #i = index(ibcc_user_list,u)
    #print i,animal_index,found
    f.write(str(i)+","+str(animal_index)+","+str(found)+"\n")

f.close()
print real_animals,fake_animals
prior = real_animals/float(real_animals + fake_animals)

print true_pos,false_pos
print true_neg,false_neg

confusion = [[max(int(true_neg),1),max(int(false_pos),1)],[max(int(false_neg),1),max(int(true_pos),1)]]

with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array(["+str(max(int((1-prior)*100),1))+","+str(max(int(prior*100),1))+"])\n")
    f.write("alpha0 = np.array("+str(confusion)+")\n")

#start by removing all temp files
try:
    os.remove(base_directory+"/Databases/condor_ibcc.out")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.csv.dat")
except OSError:
    pass

#pickle.dump((big_subjectList,big_userList),open(base_directory+"/Databases/tempOut.pickle","wb"))
ibcc.runIbcc(base_directory+"/Databases/condor_ibcc.py")
ibcc_v = []
with open(base_directory+"/Databases/condor_ibcc.out","rb") as f:
    ibcc_results = csv.reader(f, delimiter=' ')

    for row in ibcc_results:
        ibcc_v.append(float(row[2]))

with open(base_directory+"/Databases/condor_ibcc.mat","rb") as f:
    ibcc_results = csv.reader(f, delimiter=' ')

    for row in ibcc_results:
        ibcc_v.append(float(row[2]))

X_positive = []
X_negative = []

for ii,zooniverse_id in enumerate(results_dict):
    print zooniverse_id
    #print goldStandard[zooniverse_id][1]

    for center,animal_index,users_l,user_count in results_dict[zooniverse_id]:
        distances = [dist(center,g) for g in goldStandard[zooniverse_id][0]]
        smallest_dist = min(distances)
        if smallest_dist >= 2:
            print smallest_dist
            print center
            print goldStandard[zooniverse_id][0]
        #assert smallest_dist < 2

        match_index = distances.index(smallest_dist)

        if goldStandard[zooniverse_id][1][match_index]:
            X_positive.append(ibcc_v[animal_index])
        else:
            X_negative.append(ibcc_v[animal_index])

alpha_list = X_negative[:]
alpha_list.extend(X_positive)
alpha_list.sort()

roc_X = []
roc_Y = []
for alpha in alpha_list:
    positive_count = sum([1 for x in X_positive if x >= alpha])
    positive_rate = positive_count/float(len(X_positive))

    negative_count = sum([1 for x in X_negative if x >= alpha])
    negative_rate = negative_count/float(len(X_negative))

    roc_X.append(negative_rate)
    roc_Y.append(positive_rate)



#print roc_X

plt.plot(roc_X,roc_Y,color="blue")

plt.xlim((0,1.05))
plt.plot((0,1),(0,1),'--')
plt.xlabel("False Positive Count")
plt.ylabel("True Positive Count")
#plt.plot([0.058],[0.875],'o')
plt.show()