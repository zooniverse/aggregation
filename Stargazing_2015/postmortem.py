#!/usr/bin/env python
__author__ = 'ggdhines'
import cPickle as pickle
from postgres_aggregation import PanoptesAPI
from cassandra.cluster import Cluster
import json
import numpy as np

# stargazing = PanoptesAPI()
# stargazing.__cleanup__()

# metadata = pickle.load(open("/tmp/metadata.pickle","rb"))

supernova_subject_ids = [11181,12036,42056,304693,441203]

# for i,m in enumerate(metadata):
#     if m is not None:
#         try:
#             if m["candidateID"] in ["FMTJ13254307-2932269","FMTJ13545986-2820019","FMTJ10310056-3658263","FMTJ14323134-1339276"]:
#                 supernova_subject_ids.add(i)
#         except KeyError:
#             continue

cluster = Cluster()
session = cluster.connect('panoptes')

identified_users = []

for subject_id in supernova_subject_ids:
    results = session.execute("SELECT user_id,user_ip from classification_model WHERE subject_id = "+str(subject_id))
    for x in results:
        if x.user_id is not None:
            identified_users.append(x.user_id)

print identified_users


for user_id in identified_users:
    individual = []
    group = []

    results = session.execute("SELECT subject_id,annotations from classification_model WHERE user_id = "+str(user_id))
    for x in results:
        if x.subject_id in supernova_subject_ids:
            continue
        #print x.annotations
        #print "---"
        results_2 = session.execute("SELECT user_id,annotations from classification_model WHERE subject_id = "+str(x.subject_id))

        temp_group = []
        for y in results_2:
            if y.user_id != user_id:
                temp_group.append(json.loads(y.annotations)[0]["value"])
            else:
                individual.append(json.loads(y.annotations)[0]["value"])

        if temp_group != []:
            group.append(np.mean(temp_group))

    print np.mean(individual)
    print np.mean(group)
    print



