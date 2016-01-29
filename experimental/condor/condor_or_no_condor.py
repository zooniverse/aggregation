#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import random

client = pymongo.MongoClient()
db = client['condor_2014-10-30']
condor_subjects = db["condor_subjects"]

image_annotations = []
contains_condors = []

blank_total = 0
consensus_blank_total = 0
total_retired = 0


for subject in condor_subjects.find({"state":"complete"}):
    counter = 0
    annotations = []



    try:

        reason = subject["metadata"]["retire_reason"]
        if reason in ["blank","blank_consensus"]:
            continue

        condor_count = 0
        for annotation_type in subject["metadata"]["counters"]:
            if "condor" in annotation_type:
                condor_count += 1
                annotations.append(1)
            else:
                annotations.append(0)

        if condor_count >= 3:
            contains_condors.append(True)
        else:
            contains_condors.append(False)

        image_annotations.append(annotations)
        total_retired += 1
    except KeyError:
        continue

print sum(contains_condors)
print total_retired
print "==="
nn = []
initial = 2
for i in range(50):
    missedCondor = 0

    for annotations,c in zip(image_annotations,contains_condors):
        users = list(np.random.choice(annotations,15))

        if (users[0:initial] == [0 for i in range(initial)]) and c:
            missedCondor += 1
    nn.append(missedCondor)

print np.mean(nn)


