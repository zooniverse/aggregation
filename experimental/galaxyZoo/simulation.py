#!/usr/bin/env python
__author__ = 'greg'
import os
import csv
import bisect
import numpy as np

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

pos0 = []
pos1 = []
pos2 = []

user0 = {}
user1 = {}
user2 = {}

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines/"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
else:
    base_directory = "/home/greg/"

data_directory = base_directory + "/Databases"

with open(data_directory+"/candels_t01_a00_positive.dat","rb") as f:
    for l in f.readlines():
        bisect.insort(pos0,l[:-1])
with open(data_directory+"/candels_t01_a01_positive.dat","rb") as f:
    for l in f.readlines():
        bisect.insort(pos1,l[:-1])
with open(data_directory+"/candels_t01_a02_positive.dat","rb") as f:
    for l in f.readlines():
        bisect.insort(pos2,l[:-1])

print "==--"
with open(data_directory+"/2015-01-18_galaxy_zoo_classifications.csv","rb") as f:
    reader = csv.reader(f)
    next(reader, None)

    for line in reader:
        subject_id = line[1]
        user_id = line[2]
        if len(line[5]) == 0:
            continue

        #print line[5]
        #assert line[5][1] == "-"
        candels_0 = int(line[5][-1])

        try:
            index(pos0,subject_id)
            #bisect.insort(user0,user_id)
            if not(subject_id in user0):
                user0[subject_id] = [candels_0]
            else:
                user0[subject_id].append(candels_0)
        except ValueError:
            pass

        try:
            index(pos1,subject_id)
            #bisect.insort(user1,user_id)
            if not(subject_id in user1):
                user1[subject_id] = [candels_0]
            else:
                user1[subject_id].append(candels_0)
        except ValueError:
            pass

        try:
            index(pos2,subject_id)
            #bisect.insort(user2,user_id)
            if not(subject_id in user2):
                user2[subject_id] = [candels_0]
            else:
                user2[subject_id].append(candels_0)
        except ValueError:
            pass

numVotes = 3

found = 0
notFound = 0
for subject_id,votes in user0.items():
    sample_votes = np.random.choice(votes,numVotes)
    dist = [sum([1 for v in sample_votes if v == i]) for i in [0,1,2]]
    if dist[0] == max(dist):
        found += 1
    else:
        notFound += 1

print found,notFound

found = 0
notFound = 0
for subject_id,votes in user1.items():
    sample_votes = np.random.choice(votes,numVotes)
    dist = [sum([1 for v in sample_votes if v == i]) for i in [0,1,2]]
    if dist[1] == max(dist):
        found += 1
    else:
        notFound += 1

print found,notFound

found = 0
notFound = 0
for subject_id,votes in user2.items():
    sample_votes = np.random.choice(votes,numVotes)
    dist = [sum([1 for v in sample_votes if v == i]) for i in [0,1,2]]
    if dist[2] == max(dist):
        found += 1
    else:
        notFound += 1

print found,notFound