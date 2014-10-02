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

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveDBSCAN import DivisiveDBSCAN

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['penguin_2014-09-30']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [5,10,15]
penguins_at = {k:[] for k in steps}

subject_index = 0
for classification in collection.find({"subjects": {"$elemMatch": {"zooniverse_id": "APZ0003aja"}}}):
    print len(classification["annotations"][1]["value"])
    print classification["annotations"][2]

