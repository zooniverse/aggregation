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
db = client['condor_2014-09-11']
collection = db["condor_classifications"]
collection2 = db["condor_subjects"]

subjects = []
users = []
classifications = []
class_count = []
blank_count = []
retiredBlanks = {}

with open("/home/greg/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \"/home/greg/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \"/home/greg/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \"/home/greg/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array([30,70])\n")
    f.write("alpha0 = np.array([[3, 1], [1,3]])\n")

with open("/home/greg/Databases/condor_ibcc.csv","wb") as f:
    f.write("a,b,c\n")

import datetime

def update(individual_classifications):
    #start by removing all temp files
    try:
        os.remove("/home/greg/Databases/condor_ibcc.out")
    except OSError:
        pass

    try:
        os.remove("/home/greg/Databases/condor_ibcc.mat")
    except OSError:
        pass

    try:
        os.remove("/home/greg/Databases/condor_ibcc.csv.dat")
    except OSError:
        pass

    with open("/home/greg/Databases/condor_ibcc.csv","a") as f:
        for u, s, b in individual_classifications:
            f.write(str(u)+","+str(s)+","+str(b)+"\n")


    print datetime.datetime.time(datetime.datetime.now())
    ibcc.runIbcc("/home/greg/Databases/condor_ibcc.py")
    print datetime.datetime.time(datetime.datetime.now())


def analyze():
    with open("/home/greg/Databases/condor_ibcc.out","rb") as f:
        reader = csv.reader(f,delimiter=" ")

        for subject_index,p0,p1 in reader:
            subject_index = int(float(subject_index))
            subject_id = subjects[subject_index]

            c = class_count[subject_index]
            if (float(p1) >= 0.99) and (c>= 1):
                if not(subject_id in retiredBlanks):
                    retiredBlanks[subject_id] = c
                    r = collection2.find_one({"zooniverse_id": subject_id})
                    url = r ["location"]["standard"]
                    print str(c) + "  ::  " + str(p1) + " -- " + str(url)



i = 0
unknownUsers = []
empty = 0
total = 0.
for r in collection.find({"$and": [{"tutorial": False},{"subjects": {"$ne": []}}]}):


    try:
        user_name = r["user_name"]
    except KeyError:
        continue

    subject_id = r["subjects"][0]["zooniverse_id"]

    if subject_id in retiredBlanks:
        continue

    if ((i%1000) == 0) and (i > 0):
        print i
        update(classifications)
        classifications = []
        analyze()

    if not(user_name in users):
        users.append(user_name)
    if not(subject_id in subjects):
        subjects.append(subject_id)
        class_count.append(0)
        blank_count.append(0)

    i += 1
    user_index = users.index(user_name)
    subject_index = subjects.index(subject_id)
    class_count[subject_index] += 1

    a = r["annotations"]
    if ("marks" in r["annotations"][-1]):
        blank = 0
    else:
        blank = 1
        blank_count[subject_index] += 1

    classifications.append((user_index,subject_index,blank))
    if i == 35000:
        break


print len(retiredBlanks)
print np.mean(retiredBlanks.values())
print np.median(retiredBlanks.values())