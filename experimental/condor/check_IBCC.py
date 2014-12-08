#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import cPickle as pickle
import bisect
import csv
import matplotlib.pyplot as plt


if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

(big_subjectList,big_userList) = pickle.load(open(base_directory+"/Databases/tempOut.pickle","rb"))

client = pymongo.MongoClient()
db = client['condor_2014-11-20']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

values = []
errors = 0
low = 0
X = []
Y = []
with open(base_directory+"/Databases/condor_ibcc.out","rb") as f:
    ibcc_results = csv.reader(f, delimiter=' ')

    for ii,row in enumerate(ibcc_results):
        if ii == 20000:
            break

        subject = subject_collection.find_one({"zooniverse_id":big_subjectList[ii]})

        if subject["state"] == "complete":
            prob = float(row[2])
            values.append(prob)

            try:
                total = float(sum(subject["metadata"]["counters"].values()))
                n = 0

                for tag in subject["metadata"]["counters"]:
                    if "condor" in tag:
                        n += subject["metadata"]["counters"][tag]

                ourProb =  n/total
                if prob > 0.95:

                    if (ourProb < 0.5):
                        print ourProb
                        print subject["location"]["standard"]
                        errors += 1
                    else:
                        low += 1
                X.append(prob)
                Y.append(ourProb)

            except KeyError:
                pass
print errors,low
plt.plot(X,Y,'.')
plt.xlim((-0.05,1.05))
plt.ylim((-0.05,1.05))
plt.xlabel("IBCC")
plt.ylabel("Percentage")
plt.show()