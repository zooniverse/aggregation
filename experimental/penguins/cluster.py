#!/usr/bin/env python
__author__ = 'greghines'
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import cPickle as pickle
import shutil
import urllib
import math


client = pymongo.MongoClient()
db = client['penguin_2014-09-19']
collection = db["penguin_classifications"]

pts = {}

def similarity(p1):
    print p1
    assert False
    return 1

for r in collection.find():
    subject_id = r["subjects"][0]["zooniverse_id"]

    if subject_id != "APZ0002jc3":
        continue

    user_ip = r["user_ip"]

    if not(user_ip in pts):
        pts[user_ip] = []

    user_index = pts.keys().index(user_ip)

    for marking_index in r["annotations"][1]["value"]:
        marking = r["annotations"][1]["value"][marking_index]
        if marking["value"] == "adult":
            pts[user_ip].append((float(marking["x"]),float(marking["y"]),user_index))


closePts = {}

u = pts.keys()[0]
for p in pts[u]:
    closePts[p] = []
    for u2 in pts.keys()[1:]:
        assert(u != u2)
        for p2 in pts[u2]:
            d =  math.sqrt((p[0]-p2[0])**2 + (p[1]-p2[1])**2)
            if d < 5:
                closePts[p].append(p2)

print closePts
print pts[u]
model = AgglomerativeClustering(n_clusters=10,linkage="average", affinity=similarity).fit(pts[u])