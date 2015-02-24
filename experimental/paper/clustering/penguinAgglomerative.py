#!/usr/bin/env python
__author__ = 'ggdhines'
from penguinAggregation import PenguinAggregation
import random
import os
import sys
import cPickle as pickle
import aggregation

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from agglomerativeClustering import Ward,TooBig

clusterAlg = Ward().__fit__

penguin = PenguinAggregation()
subject_ids = pickle.load(open(aggregation.base_directory+"/Databases/penguin_gold.pickle","rb"))

for i,subject in enumerate(random.sample(subject_ids,50)):
    #subject = "APZ000173v"
    print i,subject

    penguin.__readin_subject__(subject,users_to_skip=["caitlin.black"])
    try:
        blankImage = penguin.__cluster_subject__(subject, clusterAlg)
    except TooBig:
        print "too big"
        continue

    if not blankImage:
        penguin.__display_raw_markings__(subject)
        penguin.__display__markings__(subject)
