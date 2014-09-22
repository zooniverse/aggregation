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
__author__ = 'greg'

subjects = pickle.load(open("/home/greg/Databases/condor_ibcc.pickle","rb"))
blanks = []
notRetired = []

client = pymongo.MongoClient()
db = client['condor_2014-09-14']
collection = db["condor_subjects"]

toRetire = []
i = 0
with open("/home/greg/Databases/condor_ibcc.out","rb") as f:
    reader = csv.reader(f,delimiter=" ")

    for subject_index,p0,p1 in reader:
        if float(p1) >= 0.99:
            subject_index = int(float(subject_index))
            subject_id = subjects[subject_index]
            blanks.append(subject_id)

            r = collection.find_one({"zooniverse_id":subject_id})

            state = r["state"]
            if state in ["complete"]:
                pass
            elif state in ["active"]:


                if r["classification_count"] > 5:
                    toRetire.append(subject_id)
                    continue
                    print r["metadata"]["counters"]

                    i += 1
                    if i == 10:
                        break
            else:
                print state

print len(toRetire)
