#!/usr/bin/env python
import os
import csv

if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"

reader = csv.reader(open(baseDir+"test.csv","rU"), delimiter=",")

for i, row in enumerate(reader):
    print "\"" + row[2] + "\"\t\"" + row[1] + "\"\t\"" + row[11] + "\"\t\"" + row[3] + "\""