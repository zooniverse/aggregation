#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
import numpy
import math
import random
import csv
import scipy.stats as stats
# import numpy as np
import pylab as pl
import scipy.special as ss
from scipy.stats import beta as beta_func

# load subject data from CSV
subjects_index = {}
with open('/home/greg/Documents/subject_species_all.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        subjects_index[row[1]] = row[2]

# connect to the mongo server
client = pymongo.MongoClient()
db = client['serengeti_2015-02-22']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

blanks = 0

# for storing the time stamp of the last classification made by each user
current_sessions = {}
# how many subjects each user has classified during the current session
session_length = {}
# how many blank each user has classified during the current session
num_blanks = {}

X = []
Y = []

X2 = []
Y2 = []
X3 = []
Y3 = []

step = 25000

# scan through *ALL* classifications (add a .skip or .limit to look at subset)
for ii, classification in enumerate(classification_collection.find().skip(1000000).limit(4000000)):
    if ii % 10000 == 0:
        print ii
    if (ii % step == 0) and (ii > 0):
        X.append(ii)
        Y.append(blanks/float(step))
        blanks = 0

        t = []
        for x,y in zip(X2,Y2):
            if (x > 0.39) and (x < 0.41):
                t.append(y)

        if t != []:
            X3.append(ii)
            Y3.append(numpy.mean(t))
        X2 = []
        Y2 = []

    # use the ip address to identify the user - that way we track non-logged in users as well
    id =  classification["user_ip"]
    # what time was the classification made at?
    time = classification["updated_at"]

    # skip tutorial classifications
    if "tutorial" in classification:
        continue

    # get the id for the subject and find out whether it was retired as a blank
    zoonvierse_id= classification["subjects"][0]["zooniverse_id"]
    # you can include blank consensus as well but I found those to be pretty error prone
    #subject = subject_collection.find_one({"zooniverse_id":zoonvierse_id})
    #if subject["metadata"]["retire_reason"] in ["blank"]:
    if subjects_index[zoonvierse_id]=="blank":
        try:
            num_blanks[id] += 1
        except KeyError:
            num_blanks[id] = 1

        blanks += 1

    # increment the session length
    try:
        session_length[id] += 1
    except KeyError:
        session_length[id] = 1

    # how long has it been since that person made their previous classification
    # if it has been 10 minutes or more, count as a new session
    try:
        time_delta = time - current_sessions[id]
        if time_delta.seconds >= 60*30: # note, session length currently set to 60 minutes. Change second number to 10 for a 10 min session
            # store data and reset counters

            if session_length[id] >= 1:
                # max_classifications = max(max_classifications,session_length[id])
                X2.append(num_blanks[id]/float(session_length[id]))
                Y2.append(session_length[id])
            session_length[id] = 0
            num_blanks[id] = 0

            # del current_sessions[id]



    except KeyError:
        pass

    current_sessions[id] = time

plt.figure(1)
plt.subplot(211)
plt.plot(X,Y)
plt.subplot(212)
plt.plot(X3,Y3)
plt.show()