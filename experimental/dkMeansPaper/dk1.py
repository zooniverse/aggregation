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
from scipy.stats import beta

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
from divisiveKmeans_2 import DivisiveKmeans_2
from kMeans import KMeans
#from kMedoids import KMedoids
#from agglomerativeClustering import agglomerativeClustering
from quadTree import Node

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()

client = pymongo.MongoClient()
db = client['penguin_2014-10-22']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

count = 0

for subject_index,subject in enumerate(collection2.find({"metadata.path":{"$regex" : ".*BAILa2014a.*"}})):
    path = subject["metadata"]["path"]



    #print path

    if not("BAILa2014a" in path):
        continue

    if count == 100:
        break

    print count

    count += 1

    user_markings = []
    user_ips = []
    big_list = []

    zooniverse_id = subject["zooniverse_id"]

    for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
        ip = r["user_ip"]
        n = 0
        xy_list = []
        try:
            if isinstance(r["annotations"][1]["value"],dict):
                for marking in r["annotations"][1]["value"].values():
                    if marking["value"] in ["adult","chick"]:
                        x,y = (float(marking["x"]),float(marking["y"]))

                        if (x,y,ip) in big_list:
                            print "--"
                            continue

                        big_list.append((x,y,ip))
                        user_markings.append((x,y))
                        user_ips.append(ip)
        except KeyError:
            print r["annotations"]

    user_identified_condors,clusters,users = DivisiveKmeans(1).fit2(user_markings,user_ips,debug=True)
    #user_identified_condors,clusters,users = DivisiveKmeans_2(1).fit2(user_markings,user_ips,debug=True)
    #user_identified_condors,clusters,users = KMedoids(1).fit2(user_markings,user_ips,debug=True)
    #user_identified_condors = agglomerativeClustering(zip(user_markings,user_ips))
    quadRoot = Node(0,0,1000,750)
    for (m,u) in zip(user_markings,user_ips):
        quadRoot.__add_point__((m,u))

    quadRoot.__ward_traverse__()

    break
