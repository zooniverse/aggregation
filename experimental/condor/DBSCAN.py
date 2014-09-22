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
collection2 = db["condor_classifications"]

i = 0
for r in collection.find({"$and" : [{"state":"active"}, {"classification_count": {"$gt":5}}]}):
    zooniverse_id = r["zooniverse_id"]

    for r2 in collection2.find({"subjects": {"$elemMatch": {"zooniverse_id": zooniverse_id}}}):
        print r2

    break