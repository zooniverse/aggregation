#!/usr/bin/env python
__author__ = 'greg'
import pymongo
from aggregation import base_directory
from penguinAggregation import PenguinAggregation
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
elif os.path.exists("/Users/greg"):
    sys.path.append("/Users/greg/Code/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from divisiveKmeans import DivisiveKmeans
#from multiClickCorrect import __ibcc__2

clusterAlg = DivisiveKmeans().__fit__

penguin = PenguinAggregation()



client = pymongo.MongoClient()
db = client['penguin_2015-01-18']
collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

subjects = subject_collection.find({"metadata.path":{"$regex":"MAIVb2012a"}})
accuracy = []
num_gold =0
could_have = 0
missed = 0
false_pos = 0
overlaps = {}
#overlaps2 = []
Xt = []
Yt = []
gold_dict = {}
with open("/Users/greg/Databases/MAIVb2013_adult_RAW.csv","rb") as f:
    for lcount,(l,s) in enumerate(zip(f.readlines(),subjects)):
        if lcount == 50:
            break
        print lcount
        image_fname = l.split(",")[0]
        #print image_fname
        gold_string = l.split("\"")[1]
        gold_markings = gold_string[:-2].split(";")
        pts = [tuple(m.split(",")[:2]) for m in gold_markings]
        if len(pts) != len(list(set(pts))):
            print "Grrrrr"
        pts = list(set(pts))
        pts = [(float(x),float(y)) for (x,y) in pts]
        num_gold += len(pts)

        zooniverse_id = s["zooniverse_id"]


        width = s["metadata"]["original_size"]["width"]
        height = s["metadata"]["original_size"]["height"]

        pts = [(int(x)/(width/1000.),int(y)/(height/563.)) for (x,y) in pts]
        gold_dict[zooniverse_id] = pts[:]

        if penguin.__get_status__(zooniverse_id) != "complete":
            continue
        penguin.__readin_subject__(zooniverse_id)

        blankImage = penguin.__cluster_subject__(zooniverse_id, clusterAlg,fix_distinct_clusters=True)
        if blankImage:
            continue
        print zooniverse_id
        penguin.__display_raw_markings__(zooniverse_id)
        break
        penguin.__accuracy__(zooniverse_id,pts)

# penguin.__readin_users__()
# penguin.__signal_ibcc__()
# #penguin.__roc__(gold_dict)
# one_overlap = penguin.__off_by_one__(display=True)
# last_id = None
#
# for t in one_overlap:
#     if t[0] != last_id:
#         print "*****"
#         print "====="
#         last_id = t[0]
#     penguin.__relative_confusion__(t)
