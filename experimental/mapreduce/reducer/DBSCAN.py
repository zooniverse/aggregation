#!/usr/bin/env python
import sys
from sklearn.cluster import DBSCAN
import numpy as np

#read in the type for each of the parameters - continuous vs. discrete - we will ignore discrete values
additional_param_type = []
configFile = sys.argv[1]
with open(configFile, 'r') as conf:
    configuration = conf.read()
    exec(configuration)

param_type = ["continuous","continuous"]
param_type.extend(additional_param_type)

curr_subject = None
pts = []
# input comes from STDIN (standard input)
for line in sys.stdin:
    subject_id,v = line.split("\t")
    v = v[:-1]

    if curr_subject != subject_id:
        if curr_subject is not None:
            X = np.array(pts)
            db = DBSCAN(eps=100, min_samples=1).fit(X)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            cluster_id = db.labels_
            unique_ids = set(cluster_id)

            sys.stdout.write(subject_id+","+str(len(pts[0])))
            for id in unique_ids:
                if id == -1:
                    continue

                in_cluster = [p for i,p in enumerate(pts) if cluster_id[i] == id]
                #print in_cluster
                center = [np.mean(c) for c in zip(*in_cluster)]
                for c in center:
                    sys.stdout.write(","+ str(c))

            print

        curr_subject = subject_id

    param = v.split(",")
    gold_standard = (param[0] == "True")
    user_id = param[1]
    #treat the rest of the parameters as a multi-dimensional point
    pts.append([float(p) for i,p in enumerate(param[2:]) if param_type[i] == "continuous"])
