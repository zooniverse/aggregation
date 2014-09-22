#!/usr/bin/env python
from __future__ import print_function
import os
import csv
import sys

if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
else:
    sys.path.append("/Users/greghines/Code/pyIBCC/python")
import ibcc


if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"

species2 = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
#species = ['gazelleThomsons']
species = ['buffalo','wildebeest','zebra']
users = []
photos = []


def createConfigFile(classID):
    f = open(baseDir+"ibcc/"+str(classID)+"config.py",'wb')
    print("import numpy as np\nscores = np.array([0,1])", file=f)
    print("nScores = len(scores)", file=f)
    print("nClasses = 2",file=f)
    print("inputFile = '"+baseDir+"ibcc/"+str(classID)+".in'", file=f)
    print("outputFile =  '"+baseDir+"ibcc/"+str(classID)+".out'", file=f)
    print("confMatFile = '"+baseDir+"ibcc/"+str(classID)+".mat'", file=f)
    # if numClasses == 4:
    #     print("alpha0 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2,2, 2]])", file=f)
    #     print("nu0 = np.array([25.0, 25.0, 25.0, 1.0])", file=f)
    # elif numClasses == 2:
    #     print("alpha0 = np.array([[2, 1], [1, 2],])", file=f)
    #     print("nu0 = np.array([50.,50.])", file=f)
    # else:
    #     assert(False)
    f.close()


individualClassifications = []
reader = csv.reader(open(baseDir+"filtered20","rU"), delimiter="\t")
for userName, photoName, classification in reader:
    individualClassifications.append((userName,photoName,classification))

ibccClassifications = []

for i, s in enumerate(species):
    print(s)
    createConfigFile(i)


    f = open(baseDir+"ibcc/"+str(i)+".in",'wb')
    for userName,photoName,classification in individualClassifications:
        if classification == "[]":
            classification = []
        else:
            classification = [int(v) for v in classification[1:-1].split(",")]

        if not(userName in users):
            users.append(userName)
            userIndex = len(users)-1
        else:
            userIndex = users.index(userName)

        if not(photoName in photos):
            photos.append(photoName)
            photoIndex = len(photos)- 1
        else:
            photoIndex = photos.index(photoName)

        if i in classification:
            print(str(userIndex)+","+str(photoIndex)+",1", file=f)
        else:
            print(str(userIndex)+","+str(photoIndex)+",0", file=f)

    f.close()
    ibcc.runIbcc(baseDir+"ibcc/"+str(i)+"config.py")



    #read in the predicted classifications
    #next, read in the the experts' classifications
    ibccClassifications = [0 for p in photos]
    print("Reading in IBCC results")
    reader = csv.reader(open(baseDir+"ibcc/"+str(i)+".out", "rU"), delimiter=" ")
    next(reader, None)

    for row in reader:
        photoIndex = int(float(row[0]))
        pos = float(row[2])

        if pos >= 0.5:
            ibccClassifications[photoIndex] = 1

    mistakes = {}

    #now go back to the users input and estimate what their confusion matrices would like
    for userName,photoName,classification in individualClassifications:
        photoIndex = photos.index(photoName)

        if classification == "[]":
            classification = []
        else:
            classification = [int(v) for v in classification[1:-1].split(",")]

        if ibccClassifications[photoIndex] == 1:
            if not(i in classification) and len(classification) == 1:
                if len(classification) != 1:
                    continue

                correct = species[i]
                reported = species2[classification[0]]
                if correct == reported:
                    continue
                if not((correct,reported) in mistakes) :
                    mistakes[(correct,reported)] = 1
                else:
                    mistakes[(correct,reported)] += 1

    for (correct,incorrect) in mistakes:
        print(correct,incorrect,mistakes[(correct,incorrect)])
    continue


    #next, read in the the experts' classifications
    expertClassifications = [0 for p in photos]
    print("Reading in expert classification")
    reader = csv.reader(open(baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
    next(reader, None)

    for row in reader:
        photoName = row[2]
        photoIndex = photos.index(photoName)
        tagged = row[12]

        if s in tagged:
            expertClassifications[photoIndex] = 1





