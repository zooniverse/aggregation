#!/usr/bin/env python
__author__ = 'greghines'
import pymongo
import matplotlib.cbook as cbook
import random
import bisect


client = pymongo.MongoClient()
db = client['plankton_2015-01-01']
user_collection = db["plankton_subjects"]

max_classifications = 0
max_users = None

for user in user_collection.find():
    if not("classification_count" in user):
        continue
    if user["classification_count"] > max_classifications:
        max_classifications = user["classification_count"]
        max_users = user["name"]

    if user["classification_count"] > 50000:
        print user["name"],user["classification_count"]

print max_users
print max_classifications
