#!/usr/bin/env python
__author__ = 'greghines'
import pymongo




client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

for subject in collection2.find({"classification_count":20}):
    zooniverse_id = subject["zooniverse_id"]

    location = subject["location"]["standard"]
    if "54119bb" in location:
        print subject

#LOCKb/LOCKb2013b