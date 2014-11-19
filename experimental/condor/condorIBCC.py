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

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

client = pymongo.MongoClient()
db = client['condor_2014-11-06']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

ip_listing = []

to_sample_from = list(subject_collection.find({"classification_count":10}))

#the header for the csv input file
f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")

sampled_ids = []


for subject_count,subject in enumerate(random.sample(to_sample_from,500)):
    zooniverse_id = subject["zooniverse_id"]
    sampled_ids.append(zooniverse_id)

    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        ip = classification["user_ip"]
        if not(ip in ip_listing):
            ip_listing.append(ip)

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


            f.write(str(str(ip_listing.index(ip)))+","+str(subject_count)+","+found_condor+"\n")
        except ValueError:
            pass


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

#read in the results
# with open(base_directory+"/Databases/condor_ibcc.out","rb") as f:
#     reader = csv.reader(f,delimiter=" ")
#
#     for subject_index,p0,p1 in reader:
#         subject_index = int(float(subject_index))
#         print p1

#now repeat - but with fewer users per image
f.close()
f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")
for subject_count,zooniverse_id in enumerate(sampled_ids):
    user_ips_to_sample = []
    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        user_ips_to_sample.append(classification["user_ip"])


    sample = random.sample(user_ips_to_sample,2)
    for user_ip in sample:
        classification = classification_collection.find_one({"subjects.zooniverse_id":zooniverse_id,"user_ip":user_ip})

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


            f.write(str(str(ip_listing.index(user_ip)))+","+str(subject_count)+","+found_condor+"\n")
        except ValueError:
            pass

with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out2\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array([30,70])\n")
    f.write("alpha0 = np.array([[3, 1], [1,3]])\n")



#start by removing all temp files
try:
    os.remove(base_directory+"/Databases/condor_ibcc.out2")
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

f = open(base_directory+"/Databases/condor_ibcc.out","r")
f2 = open(base_directory+"/Databases/condor_ibcc.out2","r")
X = []
Y = []

m1 = {}
m2 = {}

for line in f.readlines():
    words = line[:-1].split(" ")
    s1 = int(float(words[0]))
    p1 = float(words[2])

    m1[s1] = p1

for line2 in f2.readlines():
    words2 = line2[:-1].split(" ")
    s2 = int(float(words2[0]))
    p2 = float(words2[2])

    m2[s2] = p2

for s in m1:
    if s in m2:
        X.append(m1[s])
        Y.append(m2[s])

plt.plot(X,Y,'.')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()

X2 = []
Y2 = []

for alpha in np.arange(0,1.01,0.05):
    pos_index = [i for i in range(len(X)) if X[i] > 0.5]
    neg_index = [i for i in range(len(X)) if X[i] <= 0.5]

    true_pos = [i for i in pos_index if Y[i] <= alpha]
    false_pos = [i for i in neg_index if Y[i] <= alpha]

    X2.append(len(true_pos)/float(len(pos_index)))
    Y2.append(len(false_pos)/float(len(neg_index)))

plt.plot(X2,Y2)
plt.show()
