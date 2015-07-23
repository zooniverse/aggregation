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
db = client['serengeti_2015-06-27']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

# for storing the time stamp of the last classification made by each user
current_sessions = {}
# how many subjects each user has classified during the current session
session_length = {}
# how many blank each user has classified during the current session
num_blanks = {}

# X - percentage of blanks per session
X = []
# Y - session length
Y = []

max_classifications = 0
total_blanks = 0
total = 0.
# scan through *ALL* classifications (add a .skip or .limit to look at subset)
#{"created_at":{"$gte":datetime.datetime(2012,12,10)}}
for ii, classification in enumerate(classification_collection.find().skip(0).limit(4000000)):
    if ii % 10000 == 0:
        print ii


    # use the ip address to identify the user - that way we track non-logged in users as well
    if "user_name" not in classification:
        continue
    #print classification["user_name"]
    # id =  classification["user_ip"]
    id_ =  classification["user_name"]
    # what time was the classification made at?
    time = classification["updated_at"]

    # skip tutorial classifications
    if "tutorial" in classification:
        continue

    total += 1

    # get the id for the subject and find out whether it was retired as a blank
    zoonvierse_id= classification["subjects"][0]["zooniverse_id"]
    # you can include blank consensus as well but I found those to be pretty error prone
    #subject = subject_collection.find_one({"zooniverse_id":zoonvierse_id})
    #if subject["metadata"]["retire_reason"] in ["blank"]:

    blank_classification = False
    for ann in classification["annotations"]:
        if "nothing" in ann:
            blank_classification = True

    if subjects_index[zoonvierse_id]=="blank":
        try:
            num_blanks[id_].append(0)
        except KeyError:
            num_blanks[id_] = [0]
    else:
        if blank_classification:
            try:
                num_blanks[id_].append(1)
            except KeyError:
                num_blanks[id_] = [1]
        else:
            try:
                num_blanks[id_].append(0)
            except KeyError:
                num_blanks[id_] = [0]

increase = 0
decrease = 0
seg = 4
for id_ in num_blanks:
    l = len(num_blanks[id_])
    n = num_blanks[id_]

    quarter = l/seg

    if quarter == 0:
        continue

    x = sum(n[:quarter+1])/float(quarter)
    y = sum(n[-quarter:])/float(quarter)

    X.append(x)
    Y.append(y)

    if x> y:
        decrease += 1
    if x<y:
        increase += 1

print increase
print decrease

meanX = numpy.mean(X)
meanY = numpy.mean(Y)

medianX = numpy.median(X)
medianY = numpy.mean(Y)

print meanX,meanY

plt.plot(X,Y,'.',color="blue")

# plt.plot([meanX,meanX],[0,1],"--",color="green")
# plt.plot([0,1],[meanY,meanY],"--",color="green")
#
# plt.plot([medianX,medianX],[0,1],"--",color="green")
# plt.plot([0,1],[medianY,medianY],"--",color="green")


plt.xlabel("first quarter classifications")
plt.ylabel("last quarter classifications")
plt.xlim((0,1))
plt.ylim((0,1))
plt.show()