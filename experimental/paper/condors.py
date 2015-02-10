#!/usr/bin/env python
__author__ = 'greg'
import os
import sys
from condorAggregation import CondorAggregation

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
elif os.path.exists("/Users/greg"):
    sys.path.append("/Users/greg/Code/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from divisiveKmeans import DivisiveKmeans
from multiClickCorrect import MultiClickCorrect

clusterAlg = DivisiveKmeans().__fit__
correctionAlg = MultiClickCorrect(overlap_threshold=1,min_cluster_size=2).__fix__

condor = CondorAggregation()

gold_subjects = condor.__get_gold_subjects__()
gold_sample = gold_subjects[:50]

for zooniverse_id in gold_sample:
    print zooniverse_id
    condor.__load_gold_standard__(zooniverse_id)
    condor.__readin_subject__(zooniverse_id)

    blankImage = condor.__cluster_subject__(zooniverse_id, clusterAlg,fix_distinct_clusters=True)#,correction_alg=correctionAlg)

condor.__readin_users__()
condor.__signal_ibcc__()
#condor.__roc__()
#condor.__display_false_positives__()

for zooniverse_id in gold_sample:
    condor.__display_nearest_neighbours__(zooniverse_id)