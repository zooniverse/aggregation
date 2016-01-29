#!/usr/bin/env python
__author__ = 'greg'
from condorAggregation import CondorAggregation
import os
import sys
import random

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from divisiveKmeans import DivisiveKmeans

clusterAlg = DivisiveKmeans().__fit__

condor = CondorAggregation()
zooniverse_id_list = random.sample(condor.__get_subjects_per_site__("ACW000177g"),40)

for i,zooniverse_id in enumerate(zooniverse_id_list):
    print i
    condor.__readin_subject__(zooniverse_id)
    blankImage = condor.__cluster_subject__(zooniverse_id, clusterAlg)

    if not blankImage:
        condor.__find_closest_neighbour__(zooniverse_id)



