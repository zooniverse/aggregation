#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import urllib
import matplotlib.cbook as cbook
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import random
import math

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-11']
classification_collection = db["condor_classifications"]

history = {}
double = 0
double_history = {}
print classification_collection.count()
entry = None
missing = 0
for classification in classification_collection.find():
    if "user_name" in classification:
        user = classification["user_name"]
    else:
        user = classification["user_ip"]
    try:
        zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    except IndexError:
        missing += 1
        continue

    if not(user in history):
        history[user] = [zooniverse_id]
    elif not(zooniverse_id in history[user]):
        history[user].append(zooniverse_id)
    else:
        double += 1
        entry = classification
        if not(user in double_history):
            double_history[user] = 1
        else:
            double_history[user] += 1

print missing
print double
print double_history
print entry