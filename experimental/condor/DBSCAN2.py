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
db = client['condor_2014-09-19']
collection = db["condor_classifications"]

i = 0
for r in collection.find({"tutorial":False}) :#{"$and" : [{"state":"active"}, {"classification_count": {"$gt":5}}]}):
    #zooniverse_id = r["zooniverse_id"]
    user_ip = r["user_ip"]


