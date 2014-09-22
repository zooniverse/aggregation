#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import cPickle as pickle
import shutil
import urllib

client = pymongo.MongoClient()
db = client['condor_2014-09-14']
collection = db["condor_classifications"]
collection2 = db["condor_subjects"]


blankCount = {}
alreadyRetired = []
errorCount = 0
active = 0
nonBlankCount = 0

import os
animals = ["condor","eagle","turkeyVulture","raven","coyote"]
for a in animals:
    folder = "/home/greg/Databases/condors/images/"+a
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

blankReasons = ["other-4","other-3","raven-2","carcassOrScale-1-raven-2","raven-12","carcassOrScale-4-raven-11","carcassOrScale-1-raven-11","carcassOrScale-1-raven-12","coyote-4","carcassOrScale-8","carcassOrScale-2-raven-17","carcassOrScale-3-raven-19","carcassOrScale-2-raven-19","raven-18","carcass-1-raven-20","raven-17","raven-21","carcassOrScale-4-raven-1","carcassOrScale-3-raven-1","carcass-2-other-1-raven-1","carcassOrScale-2-coyote-2","coyote-1-raven-1","carcassOrScale-1-coyote-2","coyote-2","raven-3","carcass-1-raven-4","carcass-2-raven-3","carcassOrScale-1-raven-7","raven-8","carcassOrScale-1-raven-9","carcassOrScale-1-raven-10","raven-7","carcass-1-raven-9","carcassOrScale-4-coyote-1","carcass-1-other-1","other-1","carcass-3-coyote-1","carcass-2-coyote-1","carcass-1-coyote-1","carcass-2-raven-13","carcass-2-raven-12","carcassOrScale-2-raven-11","carcassOrScale-2-raven-10","carcassOrScale-2-raven-12","carcass-2-raven-9","carcassOrScale-2-raven-9","carcassOrScale-2-raven-8","carcass-2-raven-2","carcass-1-raven-1","carcassOrScale-2-raven-2","carcassOrScale-1-coyote-1","carcassOrScale-1-coyote-1-raven-1","carcassOrScale-1-coyote-1","carcass-1-raven-6","carcass-2-raven-8","carcassOrScale-1-raven-6","raven-6","raven-5","carcassOrScale-2-raven-6","carcassOrScale-2-raven-4","carcass-1-raven-5","carcassOrScale-2-coyote-1","carcassOrScale-7","carcassOrScale-2-raven-15","carcassOrScale-2-raven-14","carcassOrScale-2-raven-13","raven-15","carcassOrScale-3-raven-15","carcassOrScale-3-raven-12","carcass-1-raven-15","carcassOrScale-4-raven-16","carcassOrScale-4-raven-14","carcassOrScale-1-raven-1","raven-1","carcassOrScale-6","carcass-5","coyote-1","blank","carcassOrScale-1","carcassOrScale-2","carcassOrScale-3","carcassOrScale-4","carcassOrScale-5","carcass-1","carcass-2","carcass-3","carcass-4","carcassOrScale-2-raven-1","carcass-4-coyote-1","carcass-3-raven-1","carcass-2-raven-1"]

print len(blankReasons)
for r in collection.find({"$and": [{"tutorial": False},{"subjects" : {"$elemMatch" : {"zooniverse_id" : {"$exists" : True}}}}]}):
    try:
        user_name = r["user_name"]
    except KeyError:
        continue

    subject_id = r["subjects"][0]["zooniverse_id"]
    _id = r["_id"]
    if subject_id in alreadyRetired:
        continue



    if not(subject_id in blankCount):
        blankCount[subject_id] = 0

    if ("marks" in r["annotations"][-1]):
        blank = 1
        for markings in r["annotations"][-1]["marks"].values():

            try:
                if markings["animal"] in ["condor","goldenEagle","turkeyVulture","coyote","raven"]:
                    blank = 0
                    break
                elif markings["animal"] in ["carcassOrScale","carcass","other"]:
                    continue
                else:
                    #print markings
                    errorCount += 1
            except KeyError:
                errorCount += 1
    else:
        blank = 1

    blankCount[subject_id] += blank

    if len(alreadyRetired) == 2000:
        break

    if blankCount[subject_id] == 3:
        alreadyRetired.append(subject_id)
        print len(alreadyRetired)

        r2 = collection2.find_one({"zooniverse_id":subject_id})

        try:
            reason = r2["metadata"]["retire_reason"]
            #print reason
            if not(reason in ["blank","blank_consensus"]):
                #print r2["location"]["standard"]
                tagged = r2["metadata"]["counters"].keys()
                overlap = set()
                for t in tagged:
                    for a in animals:
                        if a in t:
                            overlap.add(a)

                #print overlap
                #if ("condor" in t) or ("eagle" in t) or ("turkeyVulture" in t):
                if overlap != set():
                    url =  r2["location"]["standard"]
                    if not(os.path.isfile("/home/greg/Databases/condors/images/"+subject_id+".JPG")):
                        urllib.urlretrieve(url, "/home/greg/Databases/condors/images/"+subject_id+".JPG")


                    for a in overlap:
                        print a
                        os.symlink("/home/greg/Databases/condors/images/"+subject_id+".JPG","/home/greg/Databases/condors/images/"+a+"/"+subject_id+".JPG")

        except KeyError:
            #print " ++ " + r2["state"]
            active += 1

print len(alreadyRetired)
print errorCount
print active
print nonBlankCount