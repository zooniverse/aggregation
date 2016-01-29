#!/usr/bin/env python
from __future__ import print_function
import os
import csv

limit = 20
species = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']


userCount = {}
classifications = {}
timeStamps = {}
photos = []
users = []


if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"



reader = csv.reader(open(baseDir+"goldFiltered.csv","rU"), delimiter=",")

next(reader, None)
for i, row in enumerate(reader):
    #print(i)
    #if i >= 80000:
    #    break

    userName = row[1]
    photoName = row[2]
    timeStamp = row[4]
    tag = row[11]
    if tag == "":
        attributeList = []
    else:
        attributeList = [species.index(tag)]

    #is this the first this photo has been tagged at all?
    if not(photoName in classifications):
        classifications[photoName] = {}
        timeStamps[photoName] = {}



    #is this the first time this user has tagged this photo?
    #if not(pNode.__classifiedBy__(uNode)):
    if not(userName in classifications[photoName]):
        #have we reached the limit for this particular photo?
        #if pNode.__getNumClassifications__() == self.limit:
        if len(classifications[photoName]) == limit:
            #if so, skip this entry
            continue
        else:


            classifications[photoName][userName] = attributeList
            timeStamps[photoName][userName] = timeStamp

    else:
        #have we reached the limit for this particular photo?
        if len(classifications[photoName]) == limit:
            currentTimeStamp = timeStamps[photoName][userName]
            #if a user tags multiple animals at once, they will be recorded as separate entries
            #but HOPEFULLY with the same time stamp, so if the time stamps are the same
            #add this entry
            if timeStamp == currentTimeStamp:
                if (tag != "") and not(tag in classifications[photoName][userName]):
                    classifications[photoName][userName].extend(attributeList)
            else:
                continue
        else:
            if (tag != "") and not(tag in classifications[photoName][userName]):
                classifications[photoName][userName].extend(attributeList)
            timeStamps[photoName][userName] = timeStamp


for photoName in classifications:
    for userName in classifications[photoName]:
        #print(str(userNames.index(userName)) + "\t" + str(photoNames.index(photoName)) + "\t" + str(classifications[photoName][userName]))
        print(userName + "\t" + photoName + "\t" + str(classifications[photoName][userName]))
