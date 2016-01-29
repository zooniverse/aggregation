#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [5,10,15,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
import random
#for subject in collection2.find({"classification_count": 20}):

alreadyThere = True
user_markings = [] #{k:[] for k in steps}
user_ips = [] #{k:[] for k in steps}
zooniverse_id = "APZ0000cj9"
user_index = 0
for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
    print classification["annotations"][2]
    try:
        print len(classification["annotations"][1]["value"].keys())
    except AttributeError:
        print classification["annotations"][1]["value"]
