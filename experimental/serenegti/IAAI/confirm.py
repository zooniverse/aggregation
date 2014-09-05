#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup, speciesList
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

photos,users = setup(tau=50)
species = "wildebeest"
t = 0
for p in photos.values():
    if species in p.goldStandard:
        t += 1

print t

if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    baseDir = "/home/ggdhines/"
else:
    baseDir = "/home/greg/"

t = 0
found = set([])
notFound = set([])
photosList = []
with open(baseDir + "Downloads/expert_classifications_raw.csv","rU") as csvfile:
    goldreader = csv.reader(csvfile)
    next(goldreader, None)
    for line in goldreader:
        photoID = line[2]
        classification = line[12]
        if species == classification:
            found.add(photoID)
        else:
            notFound.add(photoID)

t = 0
for p in list(found):
    if p in notFound:
        t += 1

print t