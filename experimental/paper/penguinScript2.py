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
zooniverse_id_list = penguin.__get_subjects_per_site__("APZ0001x3p")[0:40]

for i,zooniverse_id in enumerate(zooniverse_id_list):
    print i
    penguin.__readin_subject__(zooniverse_id)
    blankImage = penguin.__cluster_subject__(zooniverse_id, clusterAlg)

    if not blankImage:
        print "+--"
        penguin.__find_closest_neighbour__(zooniverse_id)

penguin.__barnes_interpolation__(zooniverse_id_list)

