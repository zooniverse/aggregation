#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import random
import os
import time
from time import mktime
from datetime import datetime,timedelta
import numpy as np
from scipy.stats import ks_2samp

project = "serengeti"
date = "2014-07-28"

# for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
    code_directory = base_directory + "/github"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"

else:
    base_directory = "/home/greg"
    code_directory = base_directory + "/github"

client = pymongo.MongoClient()
db = client[project+"_"+date]
classification_collection = db[project+"_classifications"]
subject_collection = db[project+"_subjects"]
user_collection = db[project+"_users"]

users = set()

for classification in classification_collection.find({"tutorial":{"$ne":True}}).limit(100000):
    try:
        user_name = classification["user_name"]
        if user_name != "brian-c":
            users.add(user_name)
    except KeyError:
        continue

sample_users = random.sample(list(users),min(100,len(list(users))))
#print sample_users
# for user in sample_users:
#     times = []
#     correct_blanks = []
#     false_blanks = []
#     for classification in classification_collection.find({"tutorial":{"$ne":True},"user_name":user}):
#         annotations = classification["annotations"]
#         zooniverse_id = classification["subjects"][0]["zooniverse_id"]
#         keys = [ann.keys() for ann in annotations]
#         timing_index = keys.index([u'started'])
#         timing = annotations[timing_index]
#         started_at = timing["started"]
#
#         # u'Tue, 16 Sep 2014 16:11:58 GMT'
#         classify_time = time.strptime(started_at,"%a, %d %b %Y %H:%M:%S %Z")
#         user_nothing = ["nothing"] in keys
#         times.append((datetime.fromtimestamp(mktime(classify_time)),user_nothing,zooniverse_id))
#
#     times.sort(key = lambda x:x[0])
#     for i in range(len(times)-1):
#         timeDiff = times[i+1][0]-times[i][0]
#         if timeDiff < timedelta(minutes=5):
#             reportBlank = times[i][1]
#             if reportBlank:
#                 zooniverse_id = times[i][2]
#                 subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
#
#                 try:
#                     #print subject["metadata"]["retire_reason"], [ann.keys() for ann in classification["annotations"]]
#                     gold_nothing = subject["metadata"]["retire_reason"] in ["blank","blank_consensus"]
#                 except KeyError:
#                     print subject["metadata"]
#                     continue
#
#                 if gold_nothing:
#                     correct_blanks.append(timeDiff.total_seconds())
#                 else:
#                     false_blanks.append(timeDiff.total_seconds())
#
#     if false_blanks == []:
#         continue
#     print len(correct_blanks),len(false_blanks)
#     #print correct_blanks
#     #print false_blanks
#     #print np.mean(correct_blanks),np.mean(false_blanks)
#     print ks_2samp(correct_blanks,false_blanks)
#     print

species = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']


for user in sample_users:

    times = []
    correct_blanks = []
    false_blanks = []
    u = user_collection.find_one({"name":user})
    if u["classification_count"] < 150:
        continue

    print user,u["classification_count"]

    for ii,classification in enumerate(classification_collection.find({"tutorial":{"$ne":True},"user_name":user}).limit(300)):
        print ii
        annotations = classification["annotations"]
        keys = [ann.keys() for ann in annotations]
        user_nothing = ["nothing"] in keys

        species_found = [ann["species"] for ann in annotations if "species" in ann]
        #print species_found
        zooniverse_id = classification["subjects"][0]["zooniverse_id"]
        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
        classification_count = subject["classification_count"]
        #print subject["metadata"]

        # use Ali's aggregation algorithm - simple but effective
        species_count = []
        votes = {s: 0 for s in species}
        for classification2 in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}).limit(min(25,classification_count)):
            annotations2 = classification2["annotations"]
            species_found2 = [ann["species"] for ann in annotations2 if "species" in ann]
            for s in species_found2:
                votes[s] += 1

            species_count.append(len(species_found2))

        count = np.median(species_count)
        results = sorted(votes.items(), key= lambda x:x[1],reverse=True)[0:int(count)]
        if results != []:
            results = zip(*results)[0]
        #print results


        timing_index = keys.index([u'started'])
        timing = annotations[timing_index]
        started_at = timing["started"]

        # u'Tue, 16 Sep 2014 16:11:58 GMT'
        classify_time = time.strptime(started_at,"%a, %d %b %Y %H:%M:%S %Z")

        times.append((datetime.fromtimestamp(mktime(classify_time)),species_found,results))

    times.sort(key = lambda x:x[0])
    for i in range(len(times)-1):
        timeDiff = times[i+1][0]-times[i][0]
        if timeDiff < timedelta(minutes=5) and len(times[i][2]) == 1:
            #print times[i][1],times[i][2]
            if list(times[i][1]) == list(times[i][2]):
                #print "+"
                correct_blanks.append(timeDiff.total_seconds())
            else:
                #print "-"
                false_blanks.append(timeDiff.total_seconds())

    if (false_blanks == []) or (correct_blanks == []):
        continue

    #print correct_blanks
    #print false_blanks
    #print np.mean(correct_blanks),np.mean(false_blanks)
    print len(correct_blanks),len(false_blanks)
    print user
    print ks_2samp(correct_blanks,false_blanks)
