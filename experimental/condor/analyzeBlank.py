#!/usr/bin/env python
__author__ = 'greghines'

import pymongo
import random
import os
import urllib

client = pymongo.MongoClient()
db = client['condor_2014-10-30']
condor_subjects = db["condor_subjects"]

initial_blanks = []
consensus_blanks = []

for subject in condor_subjects.find({"state":"complete"}):
    zooniverse_id = subject["zooniverse_id"]
    try:
        reason = subject["metadata"]["retire_reason"]
        if reason == "blank":
            initial_blanks.append(zooniverse_id)
        elif reason == "blank_consensus":
            consensus_blanks.append(zooniverse_id)
    except KeyError:
        pass

to_sample = random.sample(initial_blanks,100)
for zooniverse_id in to_sample:
    subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
    location = subject["location"]["standard"]
    slash_index = location.rfind("/")
    f_name = location[slash_index+1:]
    if not(os.path.isfile("/home/greg/Databases/condors/images/blank/initial/"+f_name)):
            urllib.urlretrieve ("http://www.condorwatch.org/subjects/standard/"+f_name, "/home/greg/Databases/condors/images/blank/initial/"+f_name)

to_sample = random.sample(consensus_blanks,100)
for zooniverse_id in to_sample:
    subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
    location = subject["location"]["standard"]
    slash_index = location.rfind("/")
    f_name = location[slash_index+1:]
    if not(os.path.isfile("/home/greg/Databases/condors/images/blank/consensus/"+f_name)):
            urllib.urlretrieve ("http://www.condorwatch.org/subjects/standard/"+f_name, "/home/greg/Databases/condors/images/blank/consensus/"+f_name)