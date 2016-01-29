#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import random
import os
import numpy as np

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

# users = random.sample(list(user_collection.find()),500)
#
# for u in users:
#     name = u["name"]
#     print u["classification_count"]
#
#     classifications = list(classification_collection.find({"user_name":name})) # ,"tutorial":False}))
#
#     print len(classifications)
#
#     for c in []:
#         zooniverse_id = c["subjects"][0]["zooniverse_id"]
#         print zooniverse_id
#         subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
#
#     print "---"


users = set()

for classification in classification_collection.find({"tutorial":{"$ne":True}}).limit(100000):
    try:
        users.add(classification["user_name"])
    except KeyError:
        continue

sample_users = random.sample(list(users),100)

v = []

for user in sample_users:
    false_blank = 0.
    true_something = 0.
    for classification in classification_collection.find({"user_name":user,"tutorial":{"$ne":True}}):
        zooniverse_id = classification["subjects"][0]["zooniverse_id"]
        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})

        user_nothing = ["nothing"] in [ann.keys() for ann in classification["annotations"]]
        try:
            #print subject["metadata"]["retire_reason"], [ann.keys() for ann in classification["annotations"]]
            gold_nothing = subject["metadata"]["retire_reason"] in ["blank","blank_consensus"]
        except KeyError:
            print subject["metadata"]
            continue

        # not nothing - so something is in the image
        if not gold_nothing:
            if user_nothing:
                false_blank += 1
            else:
                true_something += 1

    if true_something == 0:
        continue

    p = true_something/(false_blank+true_something)
    #print (false_blank,true_something)
    print p
    v.append(p)

print min(v),max(v)

print np.mean(v),np.median(v)