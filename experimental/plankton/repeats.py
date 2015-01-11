#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo

client = pymongo.MongoClient()
db = client['plankton_2015-01-01']
classification_collection = db["plankton_classifications"]
subject_collection = db["plankton_subjects"]
user_collection = db["plankton_users"]

finished_subjects = []

for subject in subject_collection.find({"state":"complete"}):
    zooniverse_id = subject["zooniverse_id"]
    finished_subjects.append((zooniverse_id,subject["updated_at"]))

finished_subjects.sort(key=lambda x:x[1],reverse=True)
print len(finished_subjects)
count = 0
for zooniverse_id,date in finished_subjects[:1000]:
    users_l = []
    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        if "user_name" in classification:
            users_l.append(classification["user_name"])
        else:
            users_l.append(classification["user_ip"])

    if not(len(users_l) == len(list(set(users_l)))):
        count += 1
        print count