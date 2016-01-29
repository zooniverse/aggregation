#!/usr/bin/env python
__author__ = 'greghines'
import csv
baseDir = "/Users/greghines/Databases/serengeti/"

reader = csv.reader(open(baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")

next(reader, None)

photos = {}
found = []

for row in reader:
    photoStr = row[2]
    species = row[12]

    if photoStr in photos:
        if not(species in photos[photoStr]):
            photos[photoStr].append(species)
    else:
        photos[photoStr] = [species]

print len(photos)
for p in photos:
    s = sorted(photos[p])
    if not(s in found):
        found.append(s)

print len(found)