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
from zeroFix import ZeroFix

clusterAlg = DivisiveKmeans().__fit__
fixAlg = ZeroFix().__fix__

penguin = PenguinAggregation()



client = pymongo.MongoClient()
db = client['penguin_2015-01-18']
collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

accuracy = []
numGold = []

penguin.__readin_subject__("APZ00035nr")

penguin.__display_raw_markings__("APZ00035nr")