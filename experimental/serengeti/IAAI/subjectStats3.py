#!/usr/bin/env python
__author__ = 'greghines'
import csv
import os
import pymongo
import numpy as np

import operator as op
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"


client = pymongo.MongoClient()
db = client['serengeti_2014-07-28']
collection = db["serengeti_subjects"]

i = 0
blankRetire = 0
consensusRetire = 0
blank_consensusRetire = 0
completeRetire = 0

c3 = []

for r in collection.find({"tutorial": {"$ne": True}}):
    try:
        count = int(r["classification_count"])
        reason =  r["metadata"]["retire_reason"]
    except KeyError:
        continue

    if reason == "blank":
        blankRetire += 1
    elif reason == "consensus":
        consensusRetire += 1
        c3.append(r["classification_count"])
    elif reason == "blank_consensus":
        blank_consensusRetire += 1
    else:
        completeRetire += 1
    #if i == 80000:
    #    break

print blankRetire
print consensusRetire
print blank_consensusRetire
print completeRetire

print np.mean(c3)



