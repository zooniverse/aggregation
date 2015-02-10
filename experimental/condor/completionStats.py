#!/usr/bin/env python
__author__ = 'greghines'
import pymongo


client = pymongo.MongoClient()
db = client['condor_2015-01-22']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

print subject_collection.find({"state":"complete"}).count()

for subject in subject_collection.find({"state":"complete"}).limit(40):
    print subject["metadata"]["file"]

