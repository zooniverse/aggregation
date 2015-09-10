#!/usr/bin/env python
__author__ = 'greghines'
import pymongo
import matplotlib.pyplot as plt
import math

# connect to the mongodb server
client = pymongo.MongoClient()
db = client['serengeti_2015-06-27']
subjects = db["serengeti_subjects"]
classifications = db["serengeti_classifications"]

date = None

delta_t = []

for c in classifications.find({"user_name":"aliburchard"}).limit(500):
    if date is not None:
        delta_t.append((c["created_at"] - date).seconds/60.)
    date = c["created_at"]

X_l = range(int(max(delta_t)))

plt.hist(delta_t,200)
plt.show()

sessions_l = []
for x in X_l:
    sessions_l.append(math.ceil(sum([1 for t in delta_t if t >= x])))

plt.plot(X_l,sessions_l,"-o")
plt.xlim((0,200))
plt.ylim((0,200))
plt.show()