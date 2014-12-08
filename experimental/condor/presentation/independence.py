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

v = []

for subject_index,zooniverse_id in enumerate(to_sample_from):
    if subject_index == 400:
        break
    print subject_index
    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    taken_at = subject["metadata"]["taken_at"]

    gold_classification = classification_collection.find_one({"user_name":"wreness", "subjects.zooniverse_id":zooniverse_id})
    found_condor = False

    try:
        mark_index = [ann.keys() for ann in gold_classification["annotations"]].index(["marks",])
        markings = gold_classification["annotations"][mark_index].values()[0]


        try:
            for animal in markings.values():
                animal_type = animal["animal"]
                found_condor = (animal_type == "condor")
        except KeyError:
            continue
    except ValueError:
        pass

    if found_condor:
        v.append((taken_at,1))
    else:
        v.append((taken_at,0))

t = zip(*v)[1]
print np.mean(t)

v.sort(key = lambda x:x[0])
count = 0
total = 0.
for i in range(len(v)-1):
    if v[i][1] == 1:
        total += 1
        if v[i+1][1] == 1:
            count += 1

print count/total