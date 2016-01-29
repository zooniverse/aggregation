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
import bisect

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
db = client['condor_2014-11-11']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

ip_listing = []

to_sample_from = list(subject_collection.find({"classification_count":10}))

#the header for the csv input file
f_ibcc = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f_ibcc.write("a,b,c\n")

sampled_ids = []
subject_list = []
user_list = []



for classification in classification_collection.find():
    if classification["subjects"] == []:
        continue

    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    try:
        subject_index = index(subject_list,zooniverse_id)
    except ValueError:
        bisect.insort(subject_list,zooniverse_id)
        subject_index = index(subject_list,zooniverse_id)

    user_ip = classification["user_ip"]
    try:
        user_index = index(user_list,user_ip)
    except ValueError:
        bisect.insort(user_list,user_ip)
        user_index = index(user_list,user_ip)

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

        f_ibcc.write(str(user_index)+","+str(subject_index)+","+found_condor+"\n")
    except ValueError:
        pass


f_ibcc.close()

with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array([30,70])\n")
    f.write("alpha0 = np.array([[3, 1], [1,3]])\n")



#start by removing all temp files
try:
    os.remove(base_directory+"/Databases/condor_ibcc.out")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.csv.dat")
except OSError:
    pass

ibcc.runIbcc(base_directory+"/Databases/condor_ibcc.py")
toretire = 0
f = open(base_directory+"/Databases/condor_ibcc.out","r")
for line in f.readlines():
    words = line[:-1].split(" ")
    s1 = int(float(words[0]))
    p1 = float(words[2])

    if p1 >= 0.25:
        continue
    zooniverse_id = subject_list[s1]
    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    if subject["state"] == "active":
        toretire += 1
        print zooniverse_id

#print toretire
