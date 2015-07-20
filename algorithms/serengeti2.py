#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
import numpy
import math
import random

# connect to the mongo server
client = pymongo.MongoClient()
db = client['serengeti_2015-02-22']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]




for j in range(30):
    skip = random.randint(0,20000)
    classification = classification_collection.find().skip(skip).limit(1)[0]
    ip = classification["user_ip"]
    try:
        classification_count = user_collection.find_one({"ip":ip})["classification_count"]
    except TypeError:
        continue
    total = min(100,classification_count)
    error = 0

    for classification in classification_collection.find({"user_ip":ip,"tutorial":{"$ne":True}}).limit(total):
        ann = classification["annotations"][-1]
        if ("nothing" in ann) and (ann["nothing"] == "true"):
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]

            subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
            try:
                retirement = subject["metadata"]["retire_reason"]
                if retirement not in ["blank"]:
                    error += 1
            except KeyError:
                print subject
                print

    print error/float(total)