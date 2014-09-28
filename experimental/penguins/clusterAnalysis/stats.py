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
import math
from PIL import Image

client = pymongo.MongoClient()
db = client['penguin_2014-09-19']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

pts_list = []

with open(base_directory + "/Databases/penguin_expert_adult.csv") as f:
    i = 0
    for l in f.readlines():
        i += 1
        #if i == 30:
        #    break

        zooniverse_id,gold_standard_pts = l[:-1].split("\t")
        gold_num = len(gold_standard_pts.split(";"))

        r = collection2.find_one({"zooniverse_id":zooniverse_id})

        classification_count = r["classification_count"]

        #if zooniverse_id in toSkip:
        #    print "skipping"
        #    continue

        if len(gold_standard_pts.split(";")) > 30:
            #print "too many points"
            continue


        p = []
        for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
            numPts = 0
            try:
                if isinstance(r["annotations"][1]["value"],dict):
                    for marking in r["annotations"][1]["value"].values():
                        if marking["value"] == "adult":
                            numPts += 1
                p.append(numPts)
            except KeyError:
                pass

        #plt.plot(len(gold_standard_pts.split(";")),numPts,'.',color = "blue")
        #if not(gold_num in pts_list):
        #    pts_list[gold_num] = p
        #else:
        #    pts_list[gold_num].extend(p)
        pts_list.append((gold_num,p[:]))



for i,(gold_num,p) in enumerate(pts_list):
    plo,phi = np.percentile(p,[25,75])
    print plo,p,phi
    #mi = [p-min(pts_list[p]),]
    #ma = [max(pts_list[p])-p,]
    mi = [p-plo,]
    ma = [phi-p,]
    #print p,(mi,ma)
    #assert(ma >= mi)
    plt.errorbar([i],[gold_num],yerr=[np.array(mi),np.array(ma)],fmt='o',color="blue")
plt.show()
