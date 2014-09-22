#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import cPickle as pickle

sys.path.append("/home/greg/github/pyIBCC/python")
import ibcc



client = pymongo.MongoClient()
db = client['condor_2014-09-14']
collection = db["condor_classifications"]
collection2 = db["condor_subjects"]

subjects = []
users = []
classifications = []


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

import datetime
i = 0
errorCount = 0
for r in collection.find({"$and": [{"tutorial": False},{"subjects" : {"$elemMatch" : {"zooniverse_id" : {"$exists" : True}}}}]}):
    try:
        user_name = r["user_name"]
    except KeyError:
        continue

    subject_id = r["subjects"][0]["zooniverse_id"]



    if not(user_name in users):
        users.append(user_name)
    if not(subject_id in subjects):
        subjects.append(subject_id)

    user_index = users.index(user_name)
    subject_index = subjects.index(subject_id)



    if ("marks" in r["annotations"][-1]):
        blank = 1
        for markings in r["annotations"][-1]["marks"].values():
            try:
                if markings["animal"] in ["condor","raven","goldenEagle","coyote","turkeyVulture"]:
                    blank = 0
                    break
                elif markings["animal"] in ["carcassOrScale"]:
                    continue
                else:
                    errorCount += 1
            except KeyError:
                errorCount += 1
    else:
        blank = 1

    i += 1
    #if i == 1000:
    #    break
    if (i % 5000) == 0:
        print i
    classifications.append((user_index,subject_index,blank))

print "====----"
print errorCount

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

with open("/home/greg/Databases/condor_ibcc.csv","wb") as f:
    f.write("a,b,c\n")

    for u, s, b in classifications:
        f.write(str(u)+","+str(s)+","+str(b)+"\n")


print datetime.datetime.time(datetime.datetime.now())
ibcc.runIbcc("/home/greg/Databases/condor_ibcc.py")
print datetime.datetime.time(datetime.datetime.now())

pickle.dump(subjects,open("/home/greg/Databases/condor_ibcc.pickle","wb"))