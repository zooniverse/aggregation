#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import csv
__author__ = 'greghines'


class SnapshotSerengeti:
    def __init__(self):
        if os.path.isdir("/Users/greghines/Databases/serengeti"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            self.baseDir = "/home/ggdhines/Databases/serengeti/"

    def __readUserClassifications__(self):
        photoList = {}

        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rU"), delimiter=",")

        i = 0

        next(reader, None)
        for row in reader:
            userName = row[1]
            photoName = row[2]

            if photoName in photoList:
                if userName in photoList[photoName]:
                    continue
                photoList[photoName].append(userName)
            else:
                photoList[photoName] = [userName]


        print(np.mean([len(photoList[x]) for x in photoList]))

s = SnapshotSerengeti()
s.__readUserClassifications__()