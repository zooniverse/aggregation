#!/usr/bin/env python
import pymongo
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib
import matplotlib.cbook as cbook
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.patches import Ellipse
from copy import deepcopy
__author__ = 'greghines'

client = pymongo.MongoClient()
db = client['lhc_penguin_data']
collection = db["higgs_hunter_subjects"]
collection2 = db["higgs_hunter_classifications"]

for r in collection.find({"metadata.data_type":"sim"}):
    zooniverse_id = r["zooniverse_id"]

    for r2 in collection2.find({"subjects" : {"$elemMatch": {"zooniverse_id": zooniverse_id}}}):
        for a in r2["annotations"]:
            if ("value" in a) and not(a["value"] in ["vertex", "weird"]):
                for vertex in a["value"].values():
                    print vertex

    break

