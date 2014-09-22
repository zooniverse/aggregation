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
blankRetire = []
consensusRetire = []
mixedRetire = []
numBlank = 0
numMixed = 0
numConsensus = 0
moreThan = 0
errorProbabilities = []
cutOff = 3
possibleError = 0

for r in collection.find({"tutorial": {"$ne": True}}):
    try:
        count = int(r["classification_count"])
        reason =  r["metadata"]["retire_reason"]
    except KeyError:
        continue

    if count > 25:
        moreThan += 1
        continue

    if not(reason in ["blank", "blank_consensus"]):
        try:
            b = int(r["metadata"]["counters"]["blank"] * 25/float(count))
            #print count,b
            if b >= cutOff:
                #print ncr(b,cutOff)/float(ncr(count,cutOff))
                errorProbabilities.append(ncr(b,cutOff)/float(ncr(25,cutOff)))
                possibleError += 1
            else:
                pass
                #errorProbabilities.append(1)
        except KeyError:
            pass
            #errorProbabilities.append(1)

    i += 1
    #if i == 80000:
    #    break

print possibleError
print np.mean(errorProbabilities)
print np.median(errorProbabilities)
print np.percentile(errorProbabilities,75)
print np.percentile(errorProbabilities,85)


