#!/usr/bin/env python
__author__ = 'greghines'
import csv
import os
import pymongo
import numpy as np


if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"


client = pymongo.MongoClient()
db = client['serengeti_2014-07-28']
collection = db["serengeti_subjects"]

i = 0
blankRetire = []
consensusRetire = []
mixedRetire = []
numBlank = 0
numMixed = 0
numConsensus = 0

for r in collection.find({"tutorial": {"$ne": True}}):
    try:
        count = int(r["classification_count"])
        reason =  r["metadata"]["retire_reason"]
    except KeyError:
        continue

    if reason in ["blank", "blank_consensus"]:
        numBlank += 1
        blankRetire.append(min(count,25))
    else:
        numConsensus += 1
        consensusRetire.append(min(count,25))

    if reason == "blank_consensus":
        mixedRetire.append(count-10)

    #i += 1
    #if i == 100:
    #    break

print numBlank
print numConsensus
print numMixed

print np.mean(blankRetire)
print np.mean(consensusRetire)
print np.mean(mixedRetire)
print np.median(mixedRetire)


