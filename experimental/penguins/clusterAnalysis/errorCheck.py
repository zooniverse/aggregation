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
import math

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
from divisiveDBSCAN import DivisiveDBSCAN

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

steps = [5,10,15,20]
penguins_at = {k:[] for k in steps}
alreadyThere = False
subject_index = 0
import cPickle as pickle
to_sample = pickle.load(open(base_directory+"/Databases/sample.pickle","rb"))
import random
#for subject in collection2.find({"classification_count": 20}):
for zooniverse_id in random.sample(to_sample,100):
    zooniverse_id = "APZ00039fp"
    subject = collection2.find_one({"zooniverse_id": zooniverse_id})
    subject_index += 1
    #if subject_index == 2:
    #    break
    #zooniverse_id = subject["zooniverse_id"]
    sys.stderr.write("=== " + str(subject_index) + "\n")
    sys.stderr.write(zooniverse_id+ "\n")

    alreadyThere = True
    user_markings = [] #{k:[] for k in steps}
    user_ips = [] #{k:[] for k in steps}

    user_index = 0
    totals = []
    with open("/home/greg/Databases/penguins/assumptionCheck/"+zooniverse_id+".count","wb") as f:
        for classification in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
            user_index += 1
            if user_index == 21:
                break

            per_user = []
            total = 0
            ip = classification["user_ip"]
            try:
                markings_list = classification["annotations"][1]["value"]
                if isinstance(markings_list,dict):
                    for marking in markings_list.values():
                        if marking["value"] in ["adult","chick"]:
                            x,y = (float(marking["x"]),float(marking["y"]))
                            total += 1
                            user_markings.append((x,y))
                            user_ips.append(ip)

                    f.write(str(total)+"\n")
            except (KeyError, ValueError):
                    #classification["annotations"]
                    user_index += -1


    if user_markings == []:
        continue
    user_identified_penguins,clusters,t = DivisiveDBSCAN(3).fit(user_markings,user_ips,debug =True)#,base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    sys.stderr.write(str(len(user_identified_penguins)) + "\n")
    #which users are in each cluster?
    users_in_clusters = []
    for c in clusters:
        users_in_clusters.append([])
        for p in c:
            i = user_markings.index(p)
            users_in_clusters[-1].append(user_ips[i])

    X = []
    Y = []
    data = []
    for i1 in range(len(user_identified_penguins)):
        for i2 in range(i1+1,len(user_identified_penguins)):
            #if i1 == i2:
            #    continue

            m1 = user_identified_penguins[i1]
            m2 = user_identified_penguins[i2]
            dist = math.sqrt((m1[0]-m2[0])**2+(m1[1]-m2[1])**2)
            X.append(dist)

            users1 = users_in_clusters[i1]
            users2 = users_in_clusters[i2]
            overlap = len([u for u in users1 if u in users2])
            Y.append(overlap)
            data.append((dist,overlap,(i1,i2)))

    #plt.plot(X,Y,'.')
    #plt.show()
    data.sort(key = lambda x:x[0])
    #data.sort(key = lambda x:x[1])

    url = subject["location"]["standard"]
    fName = url.split("/")[-1]
    #print "http://demo.zooniverse.org/penguins/subjects/standard/"+fName
    if not(os.path.isfile(base_directory + "/Databases/penguins/images/"+fName)):
        #urllib.urlretrieve ("http://demo.zooniverse.org/penguins/subjects/standard/"+fName, "/home/greg/Databases/penguins/images/"+fName)
        urllib.urlretrieve ("http://www.penguinwatch.org/subjects/standard/"+fName, base_directory+"/Databases/penguins/images/"+fName)

    #print "/home/greg/Databases/penguins/images/"+fName
    image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+fName)
    image = plt.imread(image_file)

    fig, ax = plt.subplots()
    im = ax.imshow(image)
    #print data2.index(0)/float(len(data2))
    t,data2,clusterPairs = zip(*data) #[overlap for dist,overlap in data]
    for nn in range(1,5):
        try:


            index1 = data2.index(nn)
            print index1

            #print data2[index1]/float(len(data2))#plt.show()
            c1,c2 = clusterPairs[index1]




            plt.plot((user_identified_penguins[c1][0],user_identified_penguins[c2][0]),(user_identified_penguins[c1][1],user_identified_penguins[c2][1]),color="blue")
            #plt.show()

        except ValueError:
            pass

    plt.show()
    #plt.savefig("/home/greg/Databases/penguins/assumptionCheck/"+zooniverse_id+".jpg")
    plt.close()
    break