#!/usr/bin/env python
__author__ = 'greghines'
from itertools import chain, combinations
import csv
import math
import os
import sys
from datetime import datetime


if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"

reader = csv.reader(open(baseDir+"2014-05-25_serengeti_classifications.csv", "rU"), delimiter=",")
next(reader, None)

resultsDict = {}
resultsList = []

for i,row in enumerate(reader):
    photoStr = row[2]
    species = row[11]

    timeList = row[10]

    if timeList == "tutorial":
        continue

    timeString = timeList.split(";")[0][:-6]
    #2010-09-10T05:17:40-05:00
    try:
        time = datetime.strptime(timeString, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        continue

    if not(photoStr in resultsDict):
        resultsList.append([time,[species]])
        resultsDict[photoStr] = len(resultsList)-1
    else:
        index = resultsDict[photoStr]
        resultsList[index][1].append(species)


print len(resultsList)
resultsList.sort(key=lambda x: x[0])
for i in range(min(50,len(resultsList))):
    print len(resultsList[i])
