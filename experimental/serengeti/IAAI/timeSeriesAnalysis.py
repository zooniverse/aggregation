#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pymongo
import datetime
import math
from scipy.stats import norm

speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
speciesSubset = ["hartebeest","wildebeest","guineaFowl","buffalo","gazelleGrants"]

if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"


photosPerLocation = {}
animalsInEachPhoto = {}
#find which location has the most gold standard data
reader = csv.reader(open("/home/ggdhines/Downloads/expert_classifications_raw.csv", "rU"), delimiter=",")
next(reader, None)
client = pymongo.MongoClient()
db = client['serengeti_2014-05-13']
collection = db["serengeti_subjects"]
for i,row in enumerate(reader):
    photoStr = row[2]
    subject = collection.find_one({"zooniverse_id":photoStr})

    coords = tuple(subject["coords"])

    if not(coords in photosPerLocation):
        photosPerLocation[coords] = [photoStr]
    else:
        photosPerLocation[coords].append(photoStr)

    species = row[12]
    if not(photoStr in animalsInEachPhoto):
        animalsInEachPhoto[photoStr] = set([species,])
    elif not(species in animalsInEachPhoto[photoStr]):
        animalsInEachPhoto[photoStr].add(species)


baseCoords = max(photosPerLocation.keys(),key = lambda x:len(photosPerLocation[x])) #[-2.4281265793851357, 34.89354783753996]
print "most number of gold standard photos is " + str(len(photosPerLocation[baseCoords]))

count = {}
for s in speciesList:
    count[s] = sum([1 for p in photosPerLocation[baseCoords] if (s in animalsInEachPhoto[p])])/float(len(photosPerLocation[baseCoords]))

top5 = sorted(count.keys(), key= lambda x:count[x],reverse=True)[0:5]
for s in top5:
    print s + " " + str(count[s])

timeStamps = {}
for subject in collection.find({"coords": baseCoords}):
    try:
        photoStr = subject["zooniverse_id"]
        timeStamps[photoStr] = (subject["metadata"]["timestamps"][0]-datetime.datetime(1970,1,1)).total_seconds()/3600.
    except IndexError:
        print subject["metadata"]
        #print "timestamp missing"

baseResults = {}
for s in speciesSubset:
    baseResults[s] = []
    for p in timeStamps:
        #if we don't have gold standard for this photo, skip it
        if not(p in animalsInEachPhoto):
            continue

        if s in animalsInEachPhoto[p]:
            baseResults[s].append(1)
        else:
            baseResults[s].append(0)

sortedTimeStamps = sorted(timeStamps.items(), key = lambda x:x[1])
deltaR = [sortedTimeStamps[i+1][1]-sortedTimeStamps[i][1] for i in range(len(sortedTimeStamps)-1)]

goldTimeStamps = {photoStr:timeStamps[photoStr] for photoStr in photosPerLocation[baseCoords]}
sortedTimeStamps = sorted(goldTimeStamps.items(), key = lambda x:x[1])

speciesList = ["hartebeest","wildebeest","guineaFowl","buffalo","gazelleGrants"]
newResults = {}
for s in speciesSubset:
    newResults[s] = []
    print s
    total = 0.
    contains = 0.

    for i in range(1,len(sortedTimeStamps)):
        previousPhoto = sortedTimeStamps[i-1][0]
        currPhoto = sortedTimeStamps[i][0]

        if not(previousPhoto in animalsInEachPhoto) or not(currPhoto in animalsInEachPhoto):
            print "missing"
            continue

        if s in animalsInEachPhoto[previousPhoto]:
            total += 1
            if s in animalsInEachPhoto[currPhoto]:
                contains += 1
                newResults[s].append(1)
            else:
                newResults[s].append(0)

    print (contains/total,total)
    #print np.var(newResults[s],ddof=1)

plt.plot(range(len(deltaR)),deltaR)
print "num photos: " + str(len(timeStamps))
print np.mean(deltaR)
print np.median(deltaR)
plt.yscale('log')
#plt.show()

print "===---"
for s in speciesList:
    se = math.sqrt(np.var(baseResults[s],ddof=1)/len(baseResults[s]) + np.var(newResults[s],ddof=1)/len(newResults[s]))
    diff = np.mean(baseResults[s]) - np.mean(newResults[s])
    W = diff/se
    print math.fabs(W)

    for alpha in np.arange(0.00005,0.3,0.00005):
        if math.fabs(W) >= norm.ppf(1-alpha/2.):
            print "alpha is " + str(alpha)
            break
    #print norm.ppf(0.975)
    #print norm.ppf(0.95)