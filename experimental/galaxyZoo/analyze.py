#!/usr/bin/env python
import csv
import os
import sys
import cPickle as pickle

candels = pickle.load(open("/home/greg/Databases/candels","rb"))
positiveCandels = []
candelNames = []

with open("/home/greg/Downloads/candels_star_artifact_bright_training.csv","rb") as f:
    reader = csv.reader(f)
    next(reader, None)

    for line in reader:
        subject_id = line[0]
        if subject_id in candels:
            positiveCandels.append(candels.index(subject_id))
            candelNames.append(subject_id)

#print len(positiveCandels)

truePos = 0
falseNeg = 0
possiblePos = 0
with open("/home/greg/Databases/galaxy_zoo_ibcc.out","rb") as f:
    reader = csv.reader(f,delimiter=" ")

    for line in reader:
        subject_index = int(float(line[0]))

        probabilities = [float(p) for p in line[1:]]
        if subject_index in positiveCandels:

            if probabilities[2] == max(probabilities):
                truePos += 1
            else:
                falseNeg += 1

        else:
            if probabilities[2] == max(probabilities):
                possiblePos += 1



for c in candelNames:
    print c

#print truePos
#print falseNeg
#print possiblePos