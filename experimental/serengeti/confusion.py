#!/usr/bin/env python
from __future__ import print_function
import os
import csv
import sys
__author__ = 'ggdhines'


expertClassifications = {}
userClassifications = {}

if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"


reader = csv.reader(open(baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
next(reader, None)

for row in reader:
    photoName = row[2]
    attribute = row[12]

    if photoName in expertClassifications:
        if not(attribute in expertClassifications[photoName]):
            expertClassifications[photoName].append(attribute)
    else:
        expertClassifications[photoName]= [attribute]

##########
##########
userCount = {}

reader = csv.reader(open(baseDir+"goldFiltered.csv", "rU"), delimiter=",")
next(reader, None)

for i, row in enumerate(reader):
    userName = row[1]
    photoName = row[2]
    timeStamp = row[4]
    attribute = row[11]

    #is this the first this photo has been tagged at all?
    if not(photoName in userClassifications):
        userClassifications[photoName] = [attribute]
    else:
        userClassifications[photoName].append(attribute)


######
######
mistakeDict = {}
speciesCount = {}

for photoName in expertClassifications:
    classification = expertClassifications[photoName]
    if len(classification) != 1:
        continue
    c = classification[0]

    u = userClassifications[photoName]
    mistakes = [t for t in u if not(t in classification)]

    if not(c in speciesCount):
        speciesCount[c] = 1
    else:
        speciesCount[c] += 1

    for t in mistakes:
        if t == "":
            continue

        if not(c in mistakeDict):
            mistakeDict[c] = {}

        if not(t in mistakeDict[c]):
            mistakeDict[c][t] = 1
        else:
            mistakeDict[c][t] += 1




for c in mistakeDict:
    print("===--")
    print(speciesCount[c])
    for m in mistakeDict[c]:
        print((c,m,mistakeDict[c][m]))
