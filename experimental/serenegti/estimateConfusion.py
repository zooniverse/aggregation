#!/usr/bin/env python
from __future__ import print_function
import os
import csv
import sys

__author__ = 'greghines'
if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"

species = "buffalo"

#next, read in the the experts' classifications
print("Reading in expert classification")
reader = csv.reader(open(baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
next(reader, None)

expertDict = {}

for row in reader:
    photoName = row[2]
    tagged = row[12]

    expertDict[photoName] = (tagged == species)

correctPos = 0
correctNeg = 0
falsePos = 0
falseNeg = 0

reader = csv.reader(open(baseDir+"filtered20","rU"), delimiter="\t")
for userName, photoName, classification in reader:
    if classification == "[]":
        classification = []
    else:
        classification = [int(v) for v in classification[1:-1].split(",")]


    if expertDict[photoName]:
        if 4 in classification:
            correctPos += 1
        else:
            falseNeg += 1
    else:
        if 4 in classification:
            falsePos += 1
        else:
            correctNeg += 1


print((correctPos,falseNeg))
print((falsePos,correctNeg))