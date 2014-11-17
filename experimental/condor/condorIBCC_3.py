#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import random
import datetime
import bisect


def run_ibcc(t):
    with open(base_directory+"/Databases/condor_ibcc_"+t+".py","wb") as f:
        f.write("import numpy as np\n")
        f.write("scores = np.array([0,1])\n")
        f.write("nScores = len(scores)\n")
        f.write("nClasses = 2\n")
        f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc_"+t+".csv\"\n")
        f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc_"+t+".out\"\n")
        f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc_"+t+".mat\"\n")
        f.write("nu0 = np.array([30,70])\n")
        f.write("alpha0 = np.array([[3, 1], [1,3]])\n")



    #start by removing all temp files
    try:
        os.remove(base_directory+"/Databases/condor_ibcc_"+t+".out")
    except OSError:
        pass

    try:
        os.remove(base_directory+"/Databases/condor_ibcc_"+t+".mat")
    except OSError:
        pass

    try:
        os.remove(base_directory+"/Databases/condor_ibcc_"+t+".csv.dat")
    except OSError:
        pass

    ibcc.runIbcc(base_directory+"/Databases/condor_ibcc_"+t+".py")


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

client = pymongo.MongoClient()
db = client['condor_2014-11-10']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]


classification_record = {}

f_gold = open(base_directory+"/Databases/condor_ibcc_gold.csv","wb")
f_gold.write("a,b,c\n")
f_sample = open(base_directory+"/Databases/condor_ibcc_sample.csv","wb")
f_sample.write("a,b,c\n")

#start by finding all subjects which received at least two classifications before Sept 15th
for classification in classification_collection.find():
    if classification["created_at"] >= datetime.datetime(2014,9,15):
        break

    if classification["subjects"] == []:
        continue
    zooniverse_id = classification["subjects"][0]["zooniverse_id"]

    if not("user_name" in classification):
        continue

    user_ip = classification["user_ip"]

    #make sure this user's annotations have animal types associated with them
    try:
        mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
        markings = classification["annotations"][mark_index].values()[0]

        found_condor = "0"
        for animal in markings.values():
            try:
                animal_type = animal["animal"]
            except KeyError:
                continue
            if animal_type == "condor":
                found_condor = "1"
                break

        #if we got this far, the user does have animal types associated with their markings
        if not(zooniverse_id in classification_record):
            classification_record[zooniverse_id] = [user_ip]
        else:
            classification_record[zooniverse_id].append(user_ip)
    except ValueError:
        pass


for zooniverse_id in classification_record.keys():
    if len(classification_record[zooniverse_id]) <= 2:
        del classification_record[zooniverse_id]




to_sample_from = [k for k in classification_record if len(classification_record[k]) >= 2]
print len(to_sample_from)
gold_users = []
sample_users = []
for subject_index,zooniverse_id in enumerate(random.sample(to_sample_from,500)):

    #choose 2 classifications at random
    sampling = random.sample(classification_record[zooniverse_id],3)

    already_done = []

    #["subjects"][0]["zooniverse_id"]
    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        if classification["created_at"] >= datetime.datetime(2014,9,15):
            continue

        user_ip = classification["user_ip"]
        if user_ip in already_done:
            continue
        else:
            already_done.append(user_ip)

        #user_index = index(ip_listing,user_ip)

        #check to see if there are any markings
        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]
        except ValueError:
            continue

        found_condor = "0"
        for animal in markings.values():
            try:
                animal_type = animal["animal"]
            except KeyError:
                continue
            if animal_type == "condor":
                found_condor = "1"
                break

        if user_ip in sampling:
            if not(user_ip in sample_users):
                sample_users.append(user_ip)
            f_sample.write(str(sample_users.index(user_ip))+","+str(subject_index)+","+found_condor+"\n")
        #else:
        if not(user_ip in gold_users):
            gold_users.append(user_ip)
        f_gold.write(str(gold_users.index(user_ip))+","+str(subject_index)+","+found_condor+"\n")


#gold standard first
f_gold.close()
run_ibcc("gold")
f_sample.close()
run_ibcc("sample")


f = open(base_directory+"/Databases/condor_ibcc_gold.out","r")
f2 = open(base_directory+"/Databases/condor_ibcc_sample.out","r")

X = []
Y = []

while True:
    line = f.readline()
    line2 = f2.readline()

    if not line:
        break

    words = line[:-1].split(" ")
    s1 = int(float(words[0]))
    p1 = float(words[2])

    X.append(p1)

    words = line2[:-1].split(" ")
    s1 = int(float(words[0]))
    p1 = float(words[2])

    Y.append(p1)

plt.plot(X,Y,'.')
plt.show()
