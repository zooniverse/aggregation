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
import math

client = pymongo.MongoClient()
db = client['penguin_2014-09-19']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

zooniverse_id = None
t = 0

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"


def matchPoints(pts_gold,pts_user):
    user_to_gold_mapping = [[] for i in range(len(pts_gold))]
    for i_user,p in enumerate(pts_user):
        minimum_distance = float("inf")
        bestPoint = None
        for i_gold,p2 in enumerate(pts_gold):
            dist = math.sqrt((p[0]-p2[0])**2+ (p[1]-p2[1])**2)
            if dist < minimum_distance:
                bestPoint = i_gold
                minimum_distance = dist

        if bestPoint is not None:
            user_to_gold_mapping[bestPoint].append(i_user)

    gold_to_user_mapping = [[] for i in range(len(pts_user))]
    for i_gold,p in enumerate(pts_gold):
        minimum_distance = float("inf")
        bestPoint = None
        for i_user,p2 in enumerate(pts_user):

            dist = math.sqrt((p[0]-p2[0])**2+ (p[1]-p2[1])**2)
            if dist < minimum_distance:
                bestPoint = i_user
                minimum_distance = dist

        if bestPoint is not None:
            gold_to_user_mapping[bestPoint].append(i_gold)

    return user_to_gold_mapping,gold_to_user_mapping


epsilon_range = np.arange(4,51,0.1)#range(4,51,1)
points_range = [1,2,3,4,5,6]

def createRanges(pts):
    range_ = {}

    for m,e in pts:

        if m in range_:
            pass
        else:
            currentRanges[minPts] = [(epsilon,epsilon)]

def dbscan_search(pts_gold,pts_user):
    #print len(pts_gold)
    #print "===="
    X_ = np.array(pts_user)
    matchings = {}
    numClusters = {}
    for epsilon in epsilon_range: #[5,10,15,20,25,30,35,40,45,50]:
        #print str(epsilon) + " : ",
        for min_num_points in points_range:
            db_ = DBSCAN(eps=epsilon, min_samples=min_num_points).fit(X_)

            labels_ = db_.labels_
            unique_labels_ = set(labels_)

            cluster_centers = []



            for k in unique_labels_:
                class_member_mask = (labels_ == k)
                xy = X_[class_member_mask]
                if k != -1:
                    xSet,ySet = zip(*list(X_[class_member_mask]))
                    x = np.mean(xSet)
                    y = np.mean(ySet)
                    cluster_centers.append((x,y))

            matchings[(epsilon,min_num_points)] = matchPoints(pts_gold,cluster_centers)
            numClusters[(epsilon,min_num_points)] = len(cluster_centers)
            #print str(len(cluster_centers)) + " ",
        #print

    goodPts = []
    for epsilon in epsilon_range: #[5,10,15,20,25,30,35,40,45,50]:
        for min_num_points in points_range:
            if abs(numClusters[(epsilon,min_num_points)] - len(pts_gold)) <= 1:
                # print (epsilon,min_num_points)
                # print matchings[(epsilon,min_num_points)][0]
                # print matchings[(epsilon,min_num_points)][1]
                #plt.plot(epsilon,min_num_points,'.',color='green')
                goodPts.append((epsilon,min_num_points))

    return goodPts




overallGoodPts = None
with open(base_directory + "/Databases/penguin_expert_adult.csv") as f:
    i = 0
    for l in f.readlines():

        userPts = []
        zooniverse_id,gold_standard_pts = l[:-1].split("\t")
        r = collection2.find_one({"zooniverse_id":zooniverse_id})
        #print zooniverse_id

        #if zooniverse_id in ["APZ0002p2o","APZ0002p33"]:
        #    continue
        classification_count = r["classification_count"]
        if classification_count < 10:
            continue
        i += 1
        if i == 3:
            break

        print zooniverse_id

        object_id= str(r["_id"])
        url = r["location"]["standard"]
        #print object_id
        scale = 2.048
        goldPts =  [(int(p.split(",")[0])/scale,int(p.split(",")[1])/scale) for p in gold_standard_pts.split(";")[:-1]]

        if not(os.path.isfile(base_directory+"Databases/penguins/images/"+object_id+".JPG")):
            urllib.urlretrieve (url, base_directory+"/Databases/penguins/images/"+object_id+".JPG")

        #image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
        #image = plt.imread(image_file)
        #fig, ax = plt.subplots()
        #im = ax.imshow(image)

        #x,y = zip(*goldPts)
        #plt.plot(x,y,'.',color='blue')


        #load volunteer classifications
        for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
            for marking in r["annotations"][1]["value"].values():
                if marking["value"] == "adult":
                    userPts.append((float(marking["x"]),float(marking["y"])))

        goodPts = dbscan_search(goldPts,userPts)
        if overallGoodPts is None:
            overallGoodPts = goodPts[:]
        else:
            #print goodPts
            overallGoodPts = [p for p in goodPts if p in overallGoodPts]
        print overallGoodPts
        #break
        # X = np.array(userPts)
        # db = DBSCAN(eps=15, min_samples=2).fit(X)
        #
        # labels = db.labels_
        # unique_labels = set(labels)
        #
        # for k in unique_labels:
        #     class_member_mask = (labels == k)
        #     xy = X[class_member_mask]
        #     if k != -1:
        #         xSet,ySet = zip(*list(X[class_member_mask]))
        #         x = np.mean(xSet)
        #         y = np.mean(ySet)
        #         plt.plot(x, y, '.', color="green")

        #plt.plot(x,y,'.',color='green')
        #plt.show()
        #break
        #t += 1
            # for p in pts.split(";")[:-1]:
            #     x,y = p.split(",")
            #     gold_standard_pts.append((int(x),int(y)))
            # break

print overallGoodPts
# print zooniverse_id
#
# r = collection2.find_one({"zooniverse_id":zooniverse_id})
# print r
#
# for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
#     print r
