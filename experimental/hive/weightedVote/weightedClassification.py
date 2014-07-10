#!/usr/bin/env python
import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    subject_zooniverse_id, classificationDistribution = line.split("\t")

    bestClassification = None
    maxWeight = 0

    #convert the distribution from string format into a mapping
    for c in classificationDistribution.split(" "):

        classification, weight = c.split(":")
        weight = float(weight)

        if weight > maxWeight:
            maxWeight = weight
            bestClassification = classification

    print subject_zooniverse_id + "\t" + bestClassification
