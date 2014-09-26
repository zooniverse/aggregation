#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import urllib
import matplotlib.cbook as cbook


client = pymongo.MongoClient()
db = client['penguin_2014-09-24']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

errorCount = 0

initial_consecutive_blanks = 5
blank_classifications = {}
urls = {}

i = 0
for r in collection.find():#{"classification_count": {"$gt": 0}}):
    assert r["annotations"][0]["key"] == "animalsPresent"
    if not(r["annotations"][0]["value"]  in ["yes","no","cant_tell"]):
        #print r["annotations"]

        errorCount += 1
        continue
    i += 1
    if (i%25000) == 0:
        print i
    zooniverse_id = r["subjects"][0]["zooniverse_id"]
    blank_image = (r["annotations"][0]["value"] != "yes")



    if not(zooniverse_id in blank_classifications):
        blank_classifications[zooniverse_id] = []
        r2 = collection2.find_one({"zooniverse_id":zooniverse_id})
        urls[zooniverse_id] = r2["metadata"]["path"]#r["subjects"][0]["location"]["standard"]

    if blank_image:
        blank_classifications[zooniverse_id].append(0)
    else:
        blank_classifications[zooniverse_id].append(1)

#print errorCount

false_blank_counter = 0
true_blank_counter = 0
total_counter = 0
nowRetire = 0
actuallyBlank = 0
notBlank =0
for zooniverse_id in blank_classifications:
    #were the first initial X classifications all blank?
    #based on the threshold variable initial_consecutive_blanks

    #read this in as the gold standard
    b = blank_classifications[zooniverse_id]
    if len(b) < 10:
        continue

    total_counter += 1


    #could we do better with a different threshold?
    if (len(b) >= initial_consecutive_blanks) and (not(1 in b[:initial_consecutive_blanks])):
        #we now think this is a blank image
        if not(1 in b[:10]):
            #and we would still think this image is blank at the end of 10 classifications
            actuallyBlank += 1
        else:
            notBlank += 1
            print urls[zooniverse_id]


#print actuallyBlank
#print notBlank
