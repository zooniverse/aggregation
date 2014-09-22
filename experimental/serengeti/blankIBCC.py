#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook

sys.path.append("/home/greg/github/pyIBCC/python")
import ibcc



client = pymongo.MongoClient()
db = client['serengeti_2014-07-28']
collection = db["serengeti_classifications"]

collection2 = db["serengeti_subjects"]

subjects = []
users = []
classifications = []
class_count = {}
blank_count = {}
retiredBlanks = {}

with open("/home/greg/Databases/serengeti_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \"/home/greg/Databases/serengeti_ibcc.csv\"\n")
    f.write("outputFile = \"/home/greg/Databases/serengeti_ibcc.out\"\n")
    f.write("confMatFile = \"/home/greg/Databases/serengeti_ibcc.mat\"\n")
    f.write("nu0 = np.array([30,70])\n")
    f.write("alpha0 = np.array([[3, 1], [1,3]])\n")

with open("/home/greg/Databases/serengeti_ibcc.csv","wb") as f:
    f.write("a,b,c\n")

import datetime

def update(individual_classifications):
    #start by removing all temp files
    try:
        os.remove("/home/greg/Databases/serengeti_ibcc.out")
    except OSError:
        pass

    try:
        os.remove("/home/greg/Databases/serengeti_ibcc.mat")
    except OSError:
        pass

    try:
        os.remove("/home/greg/Databases/serengeti_ibcc.csv.dat")
    except OSError:
        pass

    with open("/home/greg/Databases/serengeti_ibcc.csv","a") as f:
        for u, s, b in individual_classifications:
            f.write(str(u)+","+str(s)+","+str(b)+"\n")


    print datetime.datetime.time(datetime.datetime.now())
    ibcc.runIbcc("/home/greg/Databases/serengeti_ibcc.py")
    print datetime.datetime.time(datetime.datetime.now())


def analyze():
    with open("/home/greg/Databases/serengeti_ibcc.out","rb") as f:
        reader = csv.reader(f,delimiter=" ")

        for subject_index,p0,p1 in reader:
            subject_index = int(float(subject_index))
            subject_id = subjects[subject_index]

            c = class_count[subject_id]
            if (float(p1) >= 0.995) and (c>= 2):
                if not(subject_id in retiredBlanks):
                    retiredBlanks[subject_id] = c
                #print str(c) + "  ::  " + str(p1)



i = 0
unknownUsers = []
for r in collection.find({"tutorial": {"$ne": True}}):



    try:
        user_name = r["user_name"]
    except KeyError:
        unknownUsers.append(r["user_ip"])
        continue
    zooniverse_id = r["subjects"][0]["zooniverse_id"]
    if zooniverse_id in retiredBlanks:
        continue

    if ((i%10000) == 0) and (i > 0):
        print i
        update(classifications)
        classifications = []
        analyze()

    if not(user_name in users):
        users.append(user_name)
    if not(zooniverse_id in subjects):
        subjects.append(zooniverse_id)
        class_count[zooniverse_id] = 0
        blank_count[zooniverse_id] = 0

    i += 1
    user_index = users.index(user_name)
    subject_index = subjects.index(zooniverse_id)
    class_count[zooniverse_id] += 1

    a = r["annotations"]
    if not("nothing" in a[-1]):
        assert('species' in a[0])
        blank = 0
    else:
        blank = 1
        blank_count[zooniverse_id] += 1

    classifications.append((user_index,subject_index,blank))
    if i >= 300000:
        break


#print len(unknownUsers)
#print len(list(set(unknownUsers)))

tBlank = 0
fBlank = 0

speciesList = ['blank','elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
errors = {s.lower():0 for s in speciesList}

for zooniverse_id in retiredBlanks:
    r = collection2.find_one({"zooniverse_id" : zooniverse_id})
    retire_reason = r["metadata"]["retire_reason"]

    if retire_reason in ["blank", "blank_consensus"]:
        tBlank += 1
    else:
        fBlank += 1

        print zooniverse_id + " :: " + str(r["location"]["standard"][0])
        f = max(r["metadata"]["counters"].items(), key = lambda x:x[1])
        print f
        try:
            errors[f[0].lower()] += 1
            print str(blank_count[zooniverse_id]) + "/" + str(class_count[zooniverse_id])
        except KeyError:
            print "---***"

        #print str(r["metadata"]["counters"].values())
        print "==---"


print tBlank
print fBlank
print np.mean(retiredBlanks.values())
print np.median(retiredBlanks.values())

print "===---"
for s in speciesList:
    if errors[s.lower()] != 0:
        print s + " - " + str(errors[s.lower()])