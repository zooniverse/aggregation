#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import urllib
import matplotlib.cbook as cbook

client = pymongo.MongoClient()
db = client['penguin_2014-09-19']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

zooniverse_id = None
t = 0

with open("/home/greg/Databases/penguin_expert_adult.csv") as f:
    for l in f.readlines():
        zooniverse_id,gold_standard_pts = l[:-1].split("\t")
        r = collection2.find_one({"zooniverse_id":zooniverse_id})



        #find out how many classifications can be done so far - we know at least 1
        #but probably want to trim that - to say, at least 5
        classification_count = r["classification_count"]
        if classification_count >= 5:
            object_id= str(r["_id"])
            url = r["location"]["standard"]
            print object_id

            goldPts =  [(int(p.split(",")[0]),int(p.split(",")[1])) for p in gold_standard_pts.split(";")[:-1]]

            if not(os.path.isfile("/home/greg/Databases/penguins/images/"+object_id+".JPG")):
                urllib.urlretrieve (url, "/home/greg/Databases/penguins/images/"+object_id+".JPG")

            image_file = cbook.get_sample_data("/home/greg/Databases/penguins/images/"+object_id+".JPG")
            image = plt.imread(image_file)
            fig, ax = plt.subplots()
            im = ax.imshow(image)

            x,y = zip(*goldPts)
            plt.plot(x,y,'.',color='blue')

            plt.show()
            t += 1
            # for p in pts.split(";")[:-1]:
            #     x,y = p.split(",")
            #     gold_standard_pts.append((int(x),int(y)))
            # break


# print zooniverse_id
#
# r = collection2.find_one({"zooniverse_id":zooniverse_id})
# print r
#
# for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
#     print r
