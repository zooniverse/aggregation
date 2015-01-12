#!/usr/bin/env python
__author__ = 'ggdhines'
from penguinAggregation import PenguinAggregation
import random
import os
import sys

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from divisiveKmeans import DivisiveKmeans

clusterAlg = DivisiveKmeans().__fit__

penguin = PenguinAggregation()
subject_ids = penguin.__get_subjects_per_site__("APZ0001x3p")

for i,subject in enumerate(random.sample(subject_ids,50)):
    print i
    penguin.__readin_subject__(subject)
    blankImage = penguin.__cluster_subject__(subject, clusterAlg)

    if not blankImage:
        penguin.__display__markings__(subject)
        break