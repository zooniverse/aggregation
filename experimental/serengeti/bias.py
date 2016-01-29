#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo


client = pymongo.MongoClient()
db = client['serengeti_2014-07-28']
class_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

u = user_collection.find_one({"classification_count": {"$gt": 1000}})
u_name = u["name"]
species = "zebra"
ll = []
actually_ll = []
for classification in class_collection.find({"user_name": u_name}):
    annotations = classification["annotations"]
    empty = True
    species_found = False
    for ann in annotations:
        if "count" in ann:
            empty = False
            if ann["species"] == species:
                species_found = True

    if not empty:


        zooniverse_id = classification["subjects"][0]["zooniverse_id"]
        subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
        tutorial = "tutorial" in subject
        if not(tutorial):
            if (species in subject["metadata"]["counters"]) and (subject["metadata"]["counters"][species] > 4):
                actually_ll.append(True)
            else:
                actually_ll.append(False)

            ll.append(species_found)



#print ll

print sum([1 for s in ll if s])/float(len(ll))

total_0 = 0
count_0 = 0
for index in range(len(ll)):
    if (index == 0) or (ll[index-1] == False):
        total_0 += 1
        if ll[index]:
            count_0 += 1


print count_0/float(total_0)
print len(ll),total_0

print "====----"
print "====----"
print sum([1 for s in actually_ll if s])/float(len(actually_ll))
total_0 = 0
count_0 = 0
for index in range(len(actually_ll)):
    if (index == 0) or (actually_ll[index-1] == False):
        total_0 += 1
        if actually_ll[index]:
            count_0 += 1

print count_0/float(total_0)