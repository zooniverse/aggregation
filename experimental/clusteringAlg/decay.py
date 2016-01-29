#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import time
import datetime
from dateutil.parser import parse
import math

classifications_per_day = {}

client = pymongo.MongoClient()

#db = client['serengeti_2014-07-28']
#classification_collection = db["serengeti_classifications"]
#startingDate = datetime.datetime(2012,12,11)

#db = client['penguin_2014-10-22']
#classification_collection = db["penguin_classifications"]
#startingDate = None

#db = client['condor_2014-11-23']
#classification_collection = db["condor_classifications"]
#startingDate = datetime.datetime(2014,4,14)

#db = client['cyclone_center_2014-12-08']
#classification_collection = db["cyclone_center_classifications"]
#startingDate = datetime.datetime(2012,9,27)

#db = client["sea_floor_2014-12-08"]
#classification_collection = db["sea_floor_classifications"]
#startingDate = datetime.datetime(2012,9,13)

db = client["andromeda_2014-12-08"]
classification_collection = db["andromeda_classifications"]
startingDate = None


for c in classification_collection.find().limit(10000000):
    if ("tutorial" in c) and (c["tutorial"] is True):
        continue

    d = c["created_at"]

    if (startingDate is not None) and (d < startingDate):
        continue

    # try:
    #     d = parse(c["updated_at"])
    # except KeyError:
    #     #print c["annotations"][-3]
    #     d = parse(c["annotations"][-3]["started_at"])

    t = datetime.date(d.year,d.month,d.day)
    if not(t in classifications_per_day):
        classifications_per_day[t] = 1
    else:
        classifications_per_day[t] += 1


Y = zip(*sorted(classifications_per_day.items(),key = lambda x:x[0]))[1]
# for a,b in  sorted(classifications_per_day.items(),key = lambda x:x[0]):
#     if b > 2000:
#         print a,b
offset = 1#max(Y.index(max(Y)),0)
print offset
X = range(len(Y))

Y = Y[offset:]
Y_log = [math.log(y) for y in Y]
totalY = float(sum(Y))

X = X[offset:]
X_log = [math.log(x+1) for x in X]

plt.plot(X,Y)

X2 = []
Y2 = []
#plt.show()
for x_stop in [5,20]:#5,10]:#,15,20]:
    X_temp = X_log[:x_stop]
    Y_temp = Y_log[:x_stop]
    regression = np.polyfit(X_temp,Y_temp,1)
    #print X
    #print Y
    #print regression
    p = np.poly1d(regression)
    Z = [math.exp(p(x)) for x in X_log]

    Y2.append(sum(Z)/totalY)
    X2.append(sum(Y[:x_stop])/totalY)
    plt.plot(X,Z)
#plt.plot(X2,Y2,'o-')
print X2
print Y2
plt.xlim((0,200))
plt.yscale("log")
plt.show()