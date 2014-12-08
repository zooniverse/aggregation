#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import datetime
import cPickle as pickle



if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-20']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

count = 0
f = []

for ii,classification in enumerate(classification_collection.find({"user_name":"wreness","created_at":{"$lt":datetime.datetime(2014, 9, 15)}})):

    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    t = datetime.datetime(2014,1,1)
    for user_index,classification2 in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        t = max(t,classification2["created_at"])

    if t < datetime.datetime(2014,9,15):
        count +=1
        print ii
        f.append((zooniverse_id,classification["created_at"]))

    if (ii % 200) == 0:
        pickle.dump(f,open(base_directory+"/condor_gold.pickle","wb"))
print count

pickle.dump(f,open(base_directory+"/condor_gold.pickle","wb"))