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
db = client['serengeti_2014-07-28']
collection = db["serengeti_classifications"]

i= 0
emptyVotes = {}
nonEmptyVotes = {}
cleanEmpty = []
for r in collection.find():
    i +=1
    if "tutorial" in r:
        continue

    zooniverseId = r["subjects"][0]["zooniverse_id"]
    if i == 300000:
        break
    annotations = r["annotations"]
    empty = False

    for d in annotations:
        if "nothing" in d:
            empty = True
            break

    if empty:
        if not(zooniverseId in emptyVotes):
            emptyVotes[zooniverseId] = 1
        else:
            emptyVotes[zooniverseId] += 1

        if (emptyVotes[zooniverseId] == 5) and not(zooniverseId in nonEmptyVotes):
            cleanEmpty.append(zooniverseId)
    else:
        if not(zooniverseId in nonEmptyVotes):
            nonEmptyVotes[zooniverseId] = 1
        else:
            nonEmptyVotes[zooniverseId] += 1



j = 0
for zooniverseId in cleanEmpty:
    if zooniverseId in nonEmptyVotes:
        print nonEmptyVotes[zooniverseId]



