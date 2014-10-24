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
from copy import deepcopy

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from clusterCompare import metric,metric2

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
#to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
import random
#for subject in collection2.find({"classification_count": 20}):
noise_list = {k:[] for k in steps}
gold_penguin_count = []
#gold_standard = open("/home/greg/Documents/gold_standard_penguins","rb")
#for line_index, line in enumerate(gold_standard.readlines()):
#    if line_index == 40:#41:
#        break
#
#    zooniverse_id, num_markings = line.split(" ")
#    num_markings = int(num_markings[:-1])
file_out = "/Databases/penguins_vote__.pickle"
#f = open("/home/greg/Documents/new_gold","rb")

completed_subjects = []

sites_count = {}

for subject in collection2.find({"classification_count":20}):
    zooniverse_id = subject["zooniverse_id"]
    if subject["metadata"]["counters"]["animals_present"] > 10:
        path = subject["metadata"]["path"]
        slash = path.find("/")
        site = path[:slash]
        if not(site in sites_count):
            sites_count[site] = 1
        else:
            sites_count[site] += 1

for subject in collection2.find({"classification_count":20}):
    zooniverse_id = subject["zooniverse_id"]
    if subject["metadata"]["counters"]["animals_present"] > 10:
        path = subject["metadata"]["path"]
        slash = path.find("/")
        site = path[:slash]


