#!/usr/bin/env python
from __future__ import print_function
import image
import os.path
import pymongo
from subprocess import call
import numpy
import warnings
import math

client = pymongo.MongoClient()
db = client['serengeti_2014-05-13']
collection = db['serengeti_subjects']

baseDir = "/home/ggdhines/Databases/serengeti/photos/"
included = 0

blank_manhatten = []
blank_zero = []

for document in collection.find({"tutorial": {"$exists": False}, "metadata.retire_reason": 'blank'})[0:900]:
    photos = document["location"]["standard"]
    if len(photos) == 1:
        continue

    included += 1

    image_id_l = []
    for p in photos:
        i = p.rfind("/")
        image_id = str(p[i+1:])
        image_id_l.append(image_id)
        if not(os.path.isfile(baseDir+image_id)):
            #urllib.urlretrieve(p,baseDir+image_id)
            call(["aws", "s3", "cp", "s3://www.snapshotserengeti.org/subjects/standard/"+image_id, baseDir])

    manhattan_results = []
    zero_results = []

    error = False

    for index1,id1 in enumerate(image_id_l):
        for id2 in image_id_l[index1+1:]:
            #compare
            m, z = image.main(baseDir+id1, baseDir+id2)
            manhattan_results.append(m)
            zero_results.append(z)

    if manhattan_results == []:
        print("skipping here")
        continue

    m = max(manhattan_results)
    z = numpy.mean(zero_results)

    if not(math.isnan(m)) and not(math.isnan(z)):
        blank_manhatten.append(math.log(m))
        blank_zero.append(z)
    else:
        print("skipping")

print(included)
included = 0

species_manhatten = []
species_zero = []

for document in collection.find({"tutorial": {"$exists": False}, "metadata.retire_reason": {'$ne': 'blank'}})[0:530]:
    photos = document["location"]["standard"]
    if len(photos) == 1:
        continue

    included += 1

    image_id_l = []
    for p in photos:
        i = p.rfind("/")
        image_id = str(p[i+1:])
        image_id_l.append(image_id)
        if not(os.path.isfile(baseDir+image_id)):
            #urllib.urlretrieve(p,baseDir+image_id)
            call(["aws", "s3", "cp", "s3://www.snapshotserengeti.org/subjects/standard/"+image_id, baseDir])

    manhattan_results = []
    zero_results = []

    for index1,id1 in enumerate(image_id_l):
        for id2 in image_id_l[index1+1:]:
            #compare
            m,z = image.main(baseDir+id1, baseDir+id2)
            manhattan_results.append(m)
            zero_results.append(z)

    assert(manhattan_results != [])
    assert(zero_results != [])

    species_manhatten.append(math.log(max(manhattan_results)))
    species_zero.append(numpy.mean(zero_results))

print("included: " + str(included))

import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.hist(blank_manhatten, 10, normed=1, alpha=0.5)
ax0.set_xlim(0,20)
ax0.set_ylim(0,0.6)
ax1.hist(species_manhatten, 10, normed=1, facecolor='green', alpha=0.5)
ax1.set_xlim(0,20)
ax1.set_ylim(0,0.6)

plt.show()