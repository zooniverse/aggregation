#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import random
from IPy import IP
import matplotlib.pyplot as plt

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"


sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

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
db = client['condor_2014-11-20']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]


def collect_classification(zooniverse_id,separate_users=[]):
    user_markings = []
    user_list = []
    type_list = []
    found_animal = {u: False for u in separate_users}
    num_users = 0

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id": zooniverse_id})):
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

                #only add the animal if it is not a
                try:
                    animal_type = animal["animal"]
                    #if animal_type in ["condor","turkeyVulture","goldenEagle"]:
                    if not (user in separate_users):
                        user_markings.append((x,y))
                        user_list.append(user_index)
                        type_list.append(animal_type)
                    elif (animal_type == "condor"):
                        found_animal[user] = True

                except KeyError:
                    pass


        except ValueError:
            pass

    return user_markings,user_list,found_animal,animal_type


to_sample_from = list(subject_collection.find({"tutorial":{"$ne":True},"state":"complete"}))

learning_set = random.sample(to_sample_from,500)
#find every user who has classified at least 15 of these subjects
classification_history = {}
for ii,subject in enumerate(learning_set):
    print ii
    zooniverse_id = subject["zooniverse_id"]
    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if not(user in classification_history):
            classification_history[user] = [zooniverse_id]
        else:
            classification_history[user].append(zooniverse_id)

power_users = {}
# false_pos = {}
# false_neg = {}
# true_pos = {}
# true_neg = {}

for user in classification_history:
    history = classification_history[user]
    if len(history) < 15:
        continue

    #true positive, false positive, false negative, true negative
    power_users[user] = [0.,0.,0.,0.]

    print user
    for zooniverse_id in history:
        user_markings,user_list,found_animal ,animal_types= collect_classification(zooniverse_id,separate_users=[user])

        #determine whether or not all of the other users found an animal
        if user_markings != []:
            gold,gold_clusters = DivisiveKmeans(3).fit2(user_markings,user_list,debug=True)

            if gold != []:
                gold,gold_clusters = DivisiveKmeans(3).__fix__(gold,gold_clusters,user_markings,user_list,200)
        else:
            gold = []
            gold_clusters = []

        #so we have found some animals - now we need to figure out what species they are
        #look until we have found a condor
        num_users = len(set(user_list))
        gold_condor = False
        for cluster in gold_clusters:
            #the find the animal type corresponding to each pt in this cluster
            type_list = [animal_types[user_markings.index(pt)] for pt in cluster]
            #did at least half the people tag this animal and at least half of those classify it as a condor?
            if len(type_list) >= (num_users/2.):
                condor_count = [1 for a in type_list if a == "condor"]
                if sum(condor_count) >= (len(type_list)/2.):
                    gold_condor = True
                    break

        #the rest of the users did not find an animal
        if gold == []:

            if found_animal[user]:
                #false positive
                power_users[user][1] += 1
            else:
                power_users[user][3] += 1
        else:
            #the rest of the users did find an animal
            if found_animal[user]:
                power_users[user][0] += 1
            else:
                power_users[user][2] += 1

    print power_users[user]

#now test
#find all of the subjects that at least one of our power users have classified - not including our learning set
test_set = set()
for u in power_users:
    #find out if the user id is a name or ip address
    try:
        IP(u)
        classifications = classification_collection.find({"user_ip":u,"tutorial":False}).limit(50)
    except ValueError:
        classifications = classification_collection.find({"user_name":u,"tutorial":False}).limit(50)

    for c in classifications:
        try:
            if not(c["subjects"][0]["zooniverse_id"] in learning_set):
                test_set.add(c["subjects"][0]["zooniverse_id"])
        except (KeyError, IndexError):
            pass

print "///----"
assert False
test_results = {u:[0.,0.,0.,0.] for u in power_users}
for zooniverse_id in random.sample(test_set,min(50,len(test_set))):
    print zooniverse_id
    sampled_power_users = set()

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if user in power_users:
            sampled_power_users.add(user)

    print len(list(sampled_power_users))
    #start by considering single users - go through one at a time

    for user in sampled_power_users:
        user_markings,user_list,found_animal,animal_types = collect_classification(zooniverse_id,separate_users=[user])

        #determine whether or not all of the other users found an animal
        if user_markings != []:
            gold,gold_clusters = DivisiveKmeans(3).fit2(user_markings,user_list,debug=True)

            if gold != []:
                gold,gold_clusters = DivisiveKmeans(3).__fix__(gold,gold_clusters,user_markings,user_list,200)
        else:
            gold = []

        #so we have found some animals - now we need to figure out what species they are
        #look until we have found a condor
        num_users = len(set(user_list))
        gold_condor = False
        for cluster in gold_clusters:
            #the find the animal type corresponding to each pt in this cluster
            type_list = [animal_types[user_markings.index(pt)] for pt in cluster]
            #did at least half the people tag this animal and at least half of those classify it as a condor?
            #if len(type_list) >=

        #the rest of the users did not find an animal
        if gold == []:

            if found_animal[user]:
                #false positive
                test_results[user][1] += 1
            else:
                test_results[user][3] += 1
        else:
            #the rest of the users did find an animal
            if found_animal[user]:
                test_results[user][0] += 1
            else:
                test_results[user][2] += 1

X = []
Y = []
for u in power_users:
    learned = power_users[u]
    test = test_results[u]
    x = learned[3]/(learned[2]+learned[3])

    if (test[2] + test[3]) > 0:
        y = test[3]/(test[2]+test[3])
        X.append(x)
        Y.append(y)

plt.plot(X,Y,'.')
plt.show()