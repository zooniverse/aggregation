#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import urllib
import matplotlib.cbook as cbook
from PIL import Image
import matplotlib.pyplot as plt
import warnings

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveDBSCAN import DivisiveDBSCAN

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

print base_directory

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [5,10,15,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
#to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
import random
#for subject in collection2.find({"classification_count": 20}):

alreadyThere = True
user_markings = [] #{k:[] for k in steps}
user_ips = [] #{k:[] for k in steps}
zooniverse_id = "APZ0000nw3"
user_index = 0
for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
    user_index += 1
    if user_index == 21:
        break

    per_user = []

    ip = classification["user_ip"]
    try:
        markings_list = classification["annotations"][1]["value"]
        if isinstance(markings_list,dict):
            for marking in markings_list.values():
                if marking["value"] in ["adult","chick"]:
                    x,y = (float(marking["x"]),float(marking["y"]))

                    user_markings.append((x,y))
                    user_ips.append(ip)

    except (KeyError, ValueError):
            #classification["annotations"]
            user_index += -1



user_identified_penguins = DivisiveDBSCAN(3).fit(user_markings,user_ips)#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")
#penguins_at[s].append(len(user_identified_penguins))
#print str(s) + "  -  " + str(len(user_identified_penguins))

X,Y = zip(*user_identified_penguins)

subject = collection2.find_one({"zooniverse_id": zooniverse_id})
url = subject["location"]["standard"]
fName = url.split("/")[-1]
print "http://demo.zooniverse.org/penguins/subjects/standard/"+fName
if not(os.path.isfile(base_directory + "/Databases/penguins/images/"+fName)):
    #urllib.urlretrieve ("http://demo.zooniverse.org/penguins/subjects/standard/"+fName, "/home/greg/Databases/penguins/images/"+fName)
    urllib.urlretrieve ("http://www.penguinwatch.org/subjects/standard/"+fName, base_directory+"/Databases/penguins/images/"+fName)

print "/home/greg/Databases/penguins/images/"+fName
image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+fName)
image = plt.imread(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)
plt.plot(X,Y,'.')
plt.show()
