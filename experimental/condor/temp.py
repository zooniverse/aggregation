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
db = client['condor_2014-09-11']
collection = db["condor_subjects"]
collection2 = db["condor_classifications"]

def dbscan(XY,condor_label,ip_address,epsilon=90):

    X = np.array(XY)
    db = DBSCAN(eps=epsilon, min_samples=2).fit(X)

    condor_xy = []
    user_id = []
    condor_id = []

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    for k in unique_labels:
        condor_xy.append([])
        user_id.append([])
        condor_id.append([])

        for l,xy,condor,u in zip(labels,XY,condor_label,ip_address):
            if l != k:
                continue

            condor_xy[-1].append(xy)
            user_id[-1].append(u)
            condor_id[-1].append(condor)

    return condor_xy,condor_id,user_id

for r in collection.find():
    count = r["classification_count"]

    if count == 6:


        _id = r["_id"]
        print str(_id)
        if str(_id) != "534c3cc1d31eae0543000ffd":
            continue


        #if str(_id) in ["534c3cabd31eae0543000629","534c3caad31eae054300059d","534c3ca9d31eae0543000549","534c3ca4d31eae054300034b","534c3ca2d31eae0543000249","534c3ca0d31eae0543000173","534c3c9ed31eae0543000049","534c3ca7d31eae05430004a1","534c3ca9d31eae0543000543","534c3cadd31eae054300075f","534c3c9fd31eae05430000e1","534c3ca4d31eae054300030f","534c3ca5d31eae05430003a7","534c3cacd31eae05430006f1","534c3cafd31eae05430007e7","534c3cafd31eae05430007ff","534c3c9dd31eae054300001f"]:
        #    continue
        #if str(_id) != "534c3cd8d31eae05430019c9":
        #    continue

        if not(os.path.isfile("/home/greg/Databases/penguins/images/"+str(_id)+".JPG")):
            urllib.urlretrieve ("http://www.condorwatch.org/subjects/standard/"+str(_id)+".JPG", "/home/greg/Databases/penguins/images/"+str(_id)+".JPG")

        image_file = cbook.get_sample_data("/home/greg/Databases/penguins/images/"+str(_id)+".JPG")
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)

        condorXY = []
        condorID = []
        userID= []

        scale = 1.875

        for r2 in collection2.find({"subject_ids" : {"$all": [_id]}}):




            try:
                markings = r2["annotations"][3]["marks"]

                for animal in markings.values():
                    if animal["animal"] == "condor":
                        x = scale*float(animal["x"])
                        y = scale*float(animal["y"])
                        condorXY.append((x,y))
                        plt.plot(x, y, 'o', markerfacecolor="red",markeredgecolor='k', markersize=6)
                        if "label" in animal:
                            condorID.append(animal["label"])
                        else:
                            condorID.append("no label")
                        userID.append(r2["user_ip"])
            except IndexError:
                print "empty"
            except KeyError:
                print "missing animal"

        #print condorXY
        DB_xy, DB_labels,DB_ipAddress = dbscan(condorXY,condorID,userID)
        print DB_labels
        print DB_ipAddress

        #check to see if any of the clusters contain multiple points from the same user
        #if so, split
        toSplit = []
        for i,ip in enumerate(DB_ipAddress):
            if sorted(list(set(ip))) != sorted(ip):
                toSplit.append(i)

        #check to see if any of the clusters contain the same ids, if so, we need to combine
        ## - this will come later

        #check to see if we need to both split and combine any clusters - if so, things become more complicated
        toCombine = []
        skip = []
        #print DB_labels
        print "spliting " + str(toSplit)
        for index,xy in enumerate(DB_xy):

            if index == toSplit[0]:

                xSet,ySet = zip(*list(xy))
                plt.plot(xSet, ySet, 'o', markerfacecolor="green",markeredgecolor='k', markersize=6)

        plt.show()
        assert False
        for index1,labels in enumerate(DB_labels):
            for index2, labels2 in enumerate(DB_labels[index1+1:]):
                inter = [l for l in labels if (l in labels2) and (l != 'no label')]
                if inter != []:
                    toCombine.append((index1,index2+index1+1))
                    skip.extend([index1,index2+index1+1])
                    #print (i,j+1)
                    #print index1,index2+index2+1
                    assert (index2+index1+1) > index1

        endXY = []
        endLabels = []

        print "combining " + str(toCombine)
        for i in range(len(DB_xy)):
            if i in toSplit:
                xSet,ySet = zip(*list(DB_xy[i]))
                x = np.mean(xSet)
                y = np.mean(ySet)
                plt.plot(x, y, 'o', markerfacecolor="blue",markeredgecolor='k', markersize=6)

                #split this cluster:
                for e in range(100,0,-5):
                    new_xy, new_labels, new_ipAddress = dbscan(DB_xy[i],DB_labels[i],DB_ipAddress[i],epsilon=e)
                    #check to see if the split was successful
                    successfulSplit = True
                    for ip in new_ipAddress:
                        #print sorted(list(set(ip))), sorted(ip)
                        if sorted(list(set(ip))) != sorted(ip):
                            successfulSplit = False

                    if successfulSplit:
                        #print "split at " + str(e)
                        break


                if not successfulSplit:
                    print "unsuccessful split!!!"
                    break
                endXY.extend(new_xy)
                endLabels.extend(new_labels)
            elif not(i in skip):
                #print "not skipping " + str(i)
                endXY.append(DB_xy[i])
                endLabels.append(DB_labels[i])

        for (i,j) in toCombine:
            print "now combining " + str((i,j))
            xSet,ySet = zip(*list(DB_xy[i]))
            x = np.mean(xSet)
            y = np.mean(ySet)
            plt.plot(x, y, 'o', markerfacecolor="blue",markeredgecolor='k', markersize=6)
            xSet,ySet = zip(*list(DB_xy[j]))
            x = np.mean(xSet)
            y = np.mean(ySet)
            plt.plot(x, y, 'o', markerfacecolor="blue",markeredgecolor='k', markersize=6)

            temp_xy = DB_xy[i][:]
            temp_xy.extend(DB_xy[j])
            temp_labels = DB_labels[i][:]
            temp_labels.extend(DB_labels[j])
            temp_ip = DB_ipAddress[i][:]
            temp_ip.extend(DB_ipAddress[j])

            #print temp_xy
            for e in range(100,300,5):
                #print e
                new_xy, new_labels, new_ipAddress = dbscan(temp_xy,temp_labels,temp_ip,epsilon=e)
                #check to see if the split was successful
                if len(new_xy) == 1:
                    #print "combine at " + str(e)
                    break

            endXY.extend(new_xy)
            endLabels.extend(new_labels)

        for cluster in endXY:
            #print cluster
            xSet,ySet = zip(*list(cluster))
            x = np.mean(xSet)
            y = np.mean(ySet)
            plt.plot(x, y, 'o', markerfacecolor="green",markeredgecolor='k', markersize=6)

        plt.title("num users " + str(count))
        #print endLabels
        plt.show()
        break






