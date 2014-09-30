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
    sys.path.append("/Users/greghines/Code/pyIBCC/python")
from divisiveDBSCAN import DivisiveDBSCAN
#    divisiveDBSCAN.py

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['penguin_2014-09-27']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

with open(base_directory + "/Databases/penguin_expert.csv") as f:
    i = 0
    for l in f.readlines():

        user_markings = []
        user_ips = []
        zooniverse_id,gold_standard_pts = l[:-1].split("\t")
        adult_goldstandard,chick_goldstandard = gold_standard_pts.split(":")

        r = collection2.find_one({"zooniverse_id":zooniverse_id})
        object_id= str(r["_id"])
        classification_count = r["classification_count"]

        path = r["metadata"]["path"]
        url = r["location"]["standard"]
        image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"
        original_x,original_y = r["metadata"]["original_size"]["width"],r["metadata"]["original_size"]["height"]

        #print object_id

        if not("LOCKb" in path):
            continue

        for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
            ip = r["user_ip"]
            n = 0
            xy_list = []
            try:
                if isinstance(r["annotations"][1]["value"],dict):
                    for marking in r["annotations"][1]["value"].values():
                        if marking["value"] in ["adult","chick"]:
                            x,y = (float(marking["x"]),float(marking["y"]))
                            user_markings.append((x,y))
                            user_ips.append(ip)



            except KeyError:
                print r["annotations"]




        user_identified_penguins = DivisiveDBSCAN(4).fit(user_markings,user_ips)#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")
        print len(user_identified_penguins)
        #print len(adult_goldstandard.split(";"))+len(chick_goldstandard.split(";"))

        if not(os.path.isfile(image_path)):
            urllib.urlretrieve(url, image_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
            image = plt.imread(image_file)
            fig, ax = plt.subplots()
            im = ax.imshow(image)


        #calculate the scale
        if not(os.path.isfile(image_path)):
            urllib.urlretrieve(url, image_path)

        im=Image.open(image_path)
        new_x,new_y =  im.size
        scale = original_x/float(new_x)
        offset = 3
        if adult_goldstandard == "":
            goldPts = []
        else:
            goldPts =  [(int(p.split(",")[0])/scale-offset,int(p.split(",")[1])/scale-offset) for p in adult_goldstandard.split(";")]

        if chick_goldstandard != "":
            goldPts.extend([(int(p.split(",")[0])/scale-offset,int(p.split(",")[1])/scale-offset) for p in chick_goldstandard.split(";")])
        print len(goldPts)
        x,y = zip(*goldPts)
        plt.plot(x,y,'o',color='green')
        x,y = zip(*user_identified_penguins)
        plt.plot(x,y,'.',color='blue')

        plt.show()





