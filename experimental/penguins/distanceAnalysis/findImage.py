#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import cPickle as pickle
import bisect
import csv
import matplotlib.pyplot as plt
import random
import math
import urllib
import matplotlib.cbook as cbook
from scipy.stats.stats import pearsonr

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/classifier")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
    sys.path.append("/home/greg/github/reduction/experimental/classifier")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()

client = pymongo.MongoClient()
db = client['penguin_2014-10-22']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

subjects = list(collection2.find({"metadata.retire_reason":{"$ne":"blank"},"classification_count":{"$ne":0}}))

X = []
Y = []

for subject_index,subject in enumerate(random.sample(subjects,20)):
    path = subject["metadata"]["path"]
    print path
    print subject

    object_id= str(subject["_id"])
    classification_count = subject["classification_count"]

    path = subject["metadata"]["path"]
    url = subject["location"]["standard"]
    image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"

    if not(os.path.isfile(image_path)):
        urllib.urlretrieve(url, image_path)

    image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    image = plt.imread(image_file)
    fig, ax = plt.subplots()
    im = ax.imshow(image)

    plt.show()