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
import datetime

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError



if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-20']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]


gold = pickle.load(open(base_directory+"/condor_gold.pickle","rb"))
gold.sort(key = lambda x:x[1])

to_sample_from = (zip(*gold)[0])[1301:]


big_userList = []
big_subjectList = []
animal_count = 0



#to_sample_from = list(subject_collection.find({"state":"complete"}))
#to_sample_from2 = list(subject_collection.find({"classification_count":1,"state":"active"}))

votes = []

sample = random.sample(to_sample_from,500)
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
#print [subject["zooniverse_id"] for subject in sample]
animals = ["condor","turkeyVulture","goldenEagle","raven","coyote"]

#zooniverse_list = [u'ACW00009f3', u'ACW0002spl', u'ACW0000srp', u'ACW0000vmr', u'ACW000162q', u'ACW00046vx', u'ACW0000ksg', u'ACW00007kj', u'ACW0000mh3', u'ACW0004qkl', u'ACW0000k83', u'ACW00005d6', u'ACW0001b1v', u'ACW0000dmx', u'ACW0001hea', u'ACW0000ii0', u'ACW00048k0', u'ACW0001vyr', u'ACW00004ct', u'ACW0000c90', u'ACW00015nn', u'ACW0000umz', u'ACW00001x7', u'ACW0000bxy', u'ACW00000qt', u'ACW0000u40', u'ACW0001oqg', u'ACW0000tku', u'ACW0003lrk', u'ACW0002fz4', u'ACW00017yd', u'ACW0000o8o', u'ACW0000sj7', u'ACW0000w7i', u'ACW0000wtm', u'ACW0004isy', u'ACW0000g9s', u'ACW00008vi', u'ACW0000n2y', u'ACW0000fny', u'ACW0000cas', u'ACW0000q9x', u'ACW000148e', u'ACW0000vvf', u'ACW0000piz', u'ACW000133j', u'ACW0000obu', u'ACW0000tl4', u'ACW0000kpg', u'ACW0002fi4', u'ACW0001rcw', u'ACW00015t1', u'ACW0003czs', u'ACW0003wzn', u'ACW00006us', u'ACW0000jtg', u'ACW00009xd', u'ACW0000dki', u'ACW0001wos', u'ACW00011mv', u'ACW0000rqu', u'ACW00022q0', u'ACW0000egm', u'ACW00006pi', u'ACW0000vkc', u'ACW0000n68', u'ACW00013tw', u'ACW00001rj', u'ACW0000hiw', u'ACW0000hwo', u'ACW00038ee', u'ACW0000a8s', u'ACW0000t7i', u'ACW0002avi', u'ACW0000uou', u'ACW000089l', u'ACW0000wla', u'ACW00007wj', u'ACW00019kf', u'ACW0000ihd', u'ACW00007uk', u'ACW0000gk2', u'ACW0002mvd', u'ACW0000k0r', u'ACW0002345', u'ACW0002nv4', u'ACW00013fg', u'ACW0001311', u'ACW00011am', u'ACW000048d', u'ACW0000f8f', u'ACW00008ib', u'ACW0000ccd', u'ACW00015li', u'ACW00031rs', u'ACW0000my8', u'ACW0001bb1', u'ACW0002psg', u'ACW0000pqn', u'ACW0000z26']

X = []

tp = 0.
tn = 0.
fp = 0.
fn = 0.

for subject_index,zooniverse_id in enumerate(sample):
    print subject_index

    #gold standard
    gold_classification = classification_collection.find_one({"user_name":"wreness", "subjects.zooniverse_id":zooniverse_id})
    assert gold_classification["tutorial"] == False

    gold_condor = False

    try:
        mark_index = [ann.keys() for ann in gold_classification["annotations"]].index(["marks",])
        markings = gold_classification["annotations"][mark_index].values()[0]


        try:
            for animal in markings.values():
                animal_type = animal["animal"]
                gold_condor = (animal_type == "condor")
        except KeyError:
            continue
    except ValueError:
        pass

    condor_votes = []

    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        if ("user_name" in classification) and (classification["user_name"] == "wreness"):
            continue

        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            found_condor = False

            for animal in markings.values():
                try:
                    animal_type = animal["animal"]
                    found_condor = (animal_type == "condor")
                except KeyError:
                    continue

            if found_condor:
                condor_votes.append(1)
            else:
                condor_votes.append(0)

        except (ValueError,KeyError):
            condor_votes.append(0)



    try:
        for i in range(10):
            vote_sample = random.sample(condor_votes,2)

            if np.mean(vote_sample) >= 0.5:
                if gold_condor:
                    tp += 1
                else:
                    fp += 1
            else:
                if gold_condor:
                    fn += 1
                else:
                    tn += 1

    except ValueError:
        print classification
        continue



print (tp,fp)
print (fn,tn)

