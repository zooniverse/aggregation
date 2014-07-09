#!/usr/bin/env python
__author__ = 'greghines'
import csv
import os
import pymongo


if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"


client = pymongo.MongoClient()
db = client['serengeti_2014-06-01']
collection = db["serengeti_subjects"]

baseCoords = [-2.4281265793851357, 34.89354783753996]

r = []

def __readin_gold__():
    print("Reading in expert classification")
    reader = csv.reader(open(baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
    next(reader, None)

    for row in reader:
        photoStr = row[2]
        species = row[12]

        #find the coord where this photo was taken
        subject = collection.find_one({"zooniverse_id":photoStr})
        #where was this photo taken?
        coord = subject["coords"]
        if baseCoords == coord:
            r.append((subject["metadata"]["timestamps"][0],species))


__readin_gold__()
r.sort(key=lambda x:x[0])
print r