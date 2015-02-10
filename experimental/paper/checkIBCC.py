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
from multiClickCorrect import MultiClickCorrect
correctionAlg = MultiClickCorrect(overlap_threshold=1,min_cluster_size=2).__fix__

clusterAlg = DivisiveKmeans().__fit__

penguin = PenguinAggregation()

gold_subjects = penguin.__get_gold_subjects__()
gold_sample = gold_subjects[:50]

penguin.__readin_users__()

for count,zooniverse_id in enumerate(gold_sample):
    if count == 50:
        break
    print count, zooniverse_id
    penguin.__readin_subject__(zooniverse_id,read_in_gold=True)

    blankImage = penguin.__cluster_subject__(zooniverse_id, clusterAlg,fix_distinct_clusters=True,correction_alg=correctionAlg)
    penguin.__soy_it__(zooniverse_id)


    penguin.__signal_ibcc__()
    penguin.__roc__()
# one_overlap = penguin.__off_by_one__(display=True)
# last_id = None
#
# for t in one_overlap:
#     if t[0] != last_id:
#         print "*****"
#         print "====="
#         last_id = t[0]
#     penguin.__relative_confusion__(t)
