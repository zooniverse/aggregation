#!/usr/bin/env python
__author__ = 'greghines'
import csv
import os
import pymongo
import numpy as np


client = pymongo.MongoClient()
db = client['serengeti_2014-07-28']
collection = db["serengeti_subjects"]
collection2 = db["serengeti_classifications"]

i = 0
blankRetire = []
consensusRetire = []
mixedRetire = []
numBlank = 0
numMixed = 0
numConsensus = 0
moreThan = 0
errorProbabilities = []
cutOff = 3
possibleError = 0

speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']

retired = []
errorCount = 0
threshold = 1
errors = {s:0 for s in [1,2,3,4]}
for r in collection.find({"tutorial": {"$ne": True}}):
    try:
        count = int(r["classification_count"])
        reason =  r["metadata"]["retire_reason"]
    except KeyError:
        continue

    if (reason == "consensus") or (reason == "complete"):
        id = r["zooniverse_id"]
        cleanBlank = True
        emptyCount = 0
        #s = []
        for r2 in collection2.find({"subjects": {"$elemMatch": {"zooniverse_id": id}}}).limit(min(20,count)):
            if "nothing" in r2["annotations"][-1]:
                emptyCount += 1
                #s.append(1)
                if emptyCount in [1,2,3,4]:
                    errors[emptyCount] += 1
            elif errorCount < threshold:
                cleanBlank = False
                break

        #print emptyCount,cleanBlank
        #print s
        i += 1
        print i




    if i == 500:
        break


print errors

