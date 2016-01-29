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
db = client['condor_2014-09-11']
collection = db["condor_subjects"]

i = 0
#for r in collection.find({"classification_count": {"$gte" : 1}}):
for r in collection.find({"state": "complete"}):
    i += 1
    counters = r["metadata"]["counters"]

    try:
        blankCount = counters["blank"]
    except KeyError:
        continue

    if "blank" in counters:
        print counters
    if i == 100:
        break