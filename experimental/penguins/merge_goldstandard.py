#!/usr/bin/env python
__author__ = 'greghines'
import pymongo
import re
import os
import sys

client = pymongo.MongoClient()
db = client['penguin_2014-09-27']
collection = db["penguin_subjects"]

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

adult_dict = {}
chick_dict = {}
ids = set([])


with open(base_directory + "/Databases/penguin_expert_chick.csv","rb") as f:
    for l in f.readlines():
        zooniverse_id,pts = l.split("\t")
        chick_dict[zooniverse_id] = pts[:-2]
        ids.add(zooniverse_id)

with open(base_directory + "/Databases/penguin_expert_adult.csv","rb") as f:
    for l in f.readlines():
        zooniverse_id,pts = l.split("\t")
        adult_dict[zooniverse_id] = pts[:-2]
        ids.add(zooniverse_id)

for zooniverse_id in ids:
    print zooniverse_id + "\t",
    if zooniverse_id in adult_dict:
        print adult_dict[zooniverse_id] + ":",
    else:
        print ":",

    if zooniverse_id in chick_dict:
        print chick_dict[zooniverse_id]
    else:
        print


