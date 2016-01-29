#!/usr/bin/env python
__author__ = 'greg'
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import cPickle as pickle
import shutil
import urllib
import math

def dist(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def adaptiveDBSCAN(XYpts,user_ids):
    if XYpts == []:
        return []

    pts_in_each_cluster = []
    users_in_each_cluster = []
    cluster_centers = []

    #increase the epsilon until we don't have any nearby clusters corresponding to non-overlapping
    #sets of users
    X = np.array(XYpts)
    for epsilon in [5,10,15,20,25,30]:
        db = DBSCAN(eps=epsilon, min_samples=2).fit(X)

        labels = db.labels_
        pts_in_each_cluster = []
        users_in_each_cluster = []
        cluster_centers = []

        for k in sorted(set(labels)):
            if k == -1:
                continue

            class_member_mask = (labels == k)
            pts_in_cluster = list(X[class_member_mask])
            xSet,ySet = zip(*pts_in_cluster)

            cluster_centers.append((np.mean(xSet),np.mean(ySet)))
            pts_in_each_cluster.append(pts_in_cluster[:])
            users_in_each_cluster.append([u for u,l in zip(user_ids,labels) if l == k])

        #do we have any adjacent clusters with non-overlapping sets of users
        #if so, we should merge them by increasing the epsilon value
        cluster_compare = []
        for cluster_index, (c1,users) in enumerate(zip(cluster_centers,users_in_each_cluster)):
            for cluster_index, (c2,users2) in enumerate(zip(cluster_centers[cluster_index+1:],users_in_each_cluster[cluster_index+1:])):
                overlappingUsers = [u for u in users if u in users2]
                cluster_compare.append((dist(c1,c2),overlappingUsers))

        cluster_compare.sort(key = lambda x:x[0])
        needToMerge = [] in [c[1] for c in cluster_compare[:10]]
        if not(needToMerge):
            break
    print epsilon
    print [c[1] for c in cluster_compare[:10]]
    centers_to_return = []

    #do we need to split any clusters?
    for cluster_index in range(len(cluster_centers)):
        print "splitting"
        needToSplit = (sorted(users_in_each_cluster[cluster_index]) != sorted(list(set(users_in_each_cluster[cluster_index]))))
        if needToSplit:
            subcluster_centers = []
            X = np.array(pts_in_each_cluster[cluster_index])
            for epsilon in [30,25,20,15,10,5,1,0.1,0.01]:
                db = DBSCAN(eps=epsilon, min_samples=2).fit(X)

                labels = db.labels_
                subcluster_centers = []

                needToSplit = False

                for k in sorted(set(labels)):
                    if k == -1:
                        continue

                    class_member_mask = (labels == k)
                    users_in_subcluster = [u for u,l in zip(users_in_each_cluster[cluster_index],labels) if l == k]
                    needToSplit =  (sorted(users_in_subcluster) != sorted(list(set(users_in_subcluster))))
                    if needToSplit:
                        break

                    pts_in_cluster = list(X[class_member_mask])
                    xSet,ySet = zip(*pts_in_cluster)
                    subcluster_centers.append((np.mean(xSet),np.mean(ySet)))

                if not(needToSplit):
                    break

            assert not(needToSplit)
            centers_to_return.extend(subcluster_centers)

            #if needToSplit:
            #    print pts_in_each_cluster[cluster_index]
            #    print users_in_each_cluster[cluster_index]
            #else:

        else:
            centers_to_return.append(cluster_centers[cluster_index])



    return centers_to_return


# def cluster(XYpts,user_ids):
#     if XYpts == []:
#         return []
#
#     #find out which points are noise - don't care about the actual clusters
#     needToSplit = False
#     X = np.array(XYpts)
#
#
#     #X = np.array([XYpts[i] for i in signal_pts])
#     #user_ids = [user_ids[i] for i in signal_pts]
#     oldCenters = None
#
#     needToMerge = False
#     needToSplit = False
#
#     cluster_list = []
#     usersInCluster = []
#     centers = []
#
#     for pref in [0,-100,-200,-400,-800,-1200,-2000,-2200,-2400,-2700,-3000,-3500,-4000,-5000,-6000,-10000]:
#         #now run affinity propagation to find the actual clusters
#         af = AffinityPropagation(preference=pref).fit(X)
#         #cluster_centers_indices = af.cluster_centers_indices_
#         labels = af.labels_
#
#
#
#         unique_labels = set(labels)
#
#         usersInCluster = []
#         centers = []
#         cluster_list = []
#         for k in sorted(unique_labels):
#             assert(k != -1)
#             #print k
#             usersInCluster.append([u for u,l in zip(user_ids,labels) if l == k])
#             #print XYpts
#             #print user_ids
#
#             class_member_mask = (labels == k)
#             pts_in_cluster = list(X[class_member_mask])
#             xSet,ySet = zip(*pts_in_cluster)
#             centers.append((np.mean(xSet),np.mean(ySet)))
#             cluster_list.append(pts_in_cluster[:])
#
#         compare = []
#         for cluster_index, (c1,users) in enumerate(zip(centers,usersInCluster)):
#             for cluster_index, (c2,users2) in enumerate(zip(centers[cluster_index+1:],usersInCluster[cluster_index+1:])):
#                 overlappingUsers = [u for u in users if u in users2]
#                 compare.append((dist(c1,c2),overlappingUsers))
#
#         #needToSplit = False
#         #for users in usersInCluster:
#         #    needToSplit = (sorted(users) != sorted(list(set(users))))
#         #    if needToSplit:
#         #        break
#
#         compare.sort(key = lambda x:x[0])
#
#         needToMerge = ([] in [c[1] for c in compare[:3]]) and (compare[-1][0] <= 200)
#
#         #if needToSplit:
#         #    assert(oldCenters != None)
#         #    return oldCenters
#         if not(needToMerge):
#             break
#
#         oldCenters = centers[:]
#
#     if needToMerge:
#         print compare[0:3]
#     assert not(needToMerge)
#
#     centers_to_return = []
#     for cluster_index in range(len(cluster_list)):
#         if len(list(set(usersInCluster[cluster_index]))) == 1:
#             continue
#         #split any individual cluster
#         needToSplit = (sorted(usersInCluster[cluster_index]) != sorted(list(set(usersInCluster[cluster_index]))))
#         if needToSplit:
#             #print cluster_list[cluster_index]
#             X = np.array(cluster_list[cluster_index])
#             sub_center_list = []
#             for pref in [-2400,-2200,-2000,-1200,-800,-400,-200,-100,-75,-50,-30,0]:
#                 af = AffinityPropagation(preference=pref).fit(X)
#                 #cluster_centers_indices = af.cluster_centers_indices_
#                 labels = af.labels_
#                 try:
#                     unique_labels = set(labels)
#                 except TypeError:
#                     print pref
#                     print X
#                     print usersInCluster[cluster_index]
#                     print labels
#                     raise
#                 #get the new "sub"clusters and check to see if we need to split even more
#                 for k in sorted(unique_labels):
#                     users = [u for u,l in zip(usersInCluster[cluster_index],labels) if l == k]
#                     needToSplit = (sorted(users) != sorted(list(set(users))))
#
#                     if needToSplit:
#                         break
#
#                     #add this new sub-cluster onto the list
#                     class_member_mask = (labels == k)
#                     pts_in_cluster = list(X[class_member_mask])
#                     xSet,ySet = zip(*pts_in_cluster)
#                     sub_center_list.append((np.mean(xSet),np.mean(ySet)))
#
#                 if not(needToSplit):
#                     break
#
#             #if pref == 0:
#             #    print sub_center_list
#             assert not(needToSplit)
#             #print pref
#             centers_to_return.extend([c for c in sub_center_list if len(c) > 1])
#
#
#
#         else:
#             centers_to_return.append(centers[cluster_index])
#
#     assert not(needToSplit)
#     return centers

client = pymongo.MongoClient()
db = client['penguin_2014-09-19']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

images = {}

pts = {}
ids = {}
userCount = {}
errorCount = 0
total = 0
at_5 = {}
at_10 = {}

center_5 = {}
center_10 = {}



step_1 = 5
step_2 = 8

toSkip = ["APZ0002uw3","APZ0001v9f","APZ00010ww","APZ0000p99","APZ0002jc3","APZ00014t4","APZ0000v0n","APZ0000ifx","APZ0002pch","APZ0003kls","APZ0001iv3","APZ0003auc","APZ0002ezn"]
mainSubject = "APZ0003fgt" #APZ0001jre
toPlot = None
numClassifications = []


for r in collection.find():
    subject_id = r["subjects"][0]["zooniverse_id"]
    total += 1

    if subject_id != "APZ0003kls":# in toSkip:
        continue


    if not(subject_id in pts):
        pts[subject_id] = []
        userCount[subject_id] = 0
        ids[subject_id] = []

    userCount[subject_id] += 1
    animalsPresent = r["annotations"][0]["value"] == "yes"
    #print animalsPresent
    if animalsPresent:
        c = 0
        for marking_index in r["annotations"][1]["value"]:
            try:
                marking = r["annotations"][1]["value"][marking_index]
                if True: # marking["value"] == "adult":
                    x = float(marking["x"])
                    y = float(marking["y"])
                    ip = r["user_ip"]

                    alreadyInList = False
                    try:
                        index = pts[subject_id].index((x,y))
                        if ids[subject_id][index] == ip:
                            alreadyInList = True
                    except ValueError:
                        pass

                    if not(alreadyInList):
                        pts[subject_id].append((x,y))
                        ids[subject_id].append(ip)
                        c += 1
            except TypeError:
                errorCount += 1
                userCount[subject_id] += -1
                break
            except ValueError:
                errorCount += 1
                continue

        numClassifications.append(c)



    if userCount[subject_id] in [step_2]:
        cluster_center = adaptiveDBSCAN(pts[subject_id],ids[subject_id])
        mainSubject = subject_id
        if cluster_center != []:
            break

        if userCount[subject_id] == step_1:
            pass
            #at_5[subject_id] = len(cluster_center)
        else:
            at_10[subject_id] = len(cluster_center)




# inBoth = [subject_id for subject_id in at_10 if (subject_id in at_5)]
# # print len(inBoth)
# x = [at_5[subject_id] for subject_id in inBoth]
# y = [at_10[subject_id] for subject_id in inBoth]
# print zip(inBoth,zip(x,y))
# plt.plot((0,100),(0,100),'--')
# # #print x
# # #print y
# plt.plot(x,y,'.')
# plt.show()


# print userCount
# print numClassifications
#
#
print mainSubject
r2 = collection2.find_one({"zooniverse_id":mainSubject})
url = r2["location"]["standard"]

if not(os.path.isfile("/home/greg/Databases/penguins/images/"+mainSubject+".JPG")):
    urllib.urlretrieve (url, "/home/greg/Databases/penguins/images/"+mainSubject+".JPG")

image_file = cbook.get_sample_data("/home/greg/Databases/penguins/images/"+mainSubject+".JPG")
image = plt.imread(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)
#plt.show()
#
if cluster_center != []:
    x,y = zip(*cluster_center)
    plt.plot(x,y,'.',color='blue')
#
# x,y = zip(*center_5[mainSubject])
# plt.plot(x,y,'.',color='red')
# x,y = zip(*center_10[mainSubject])
# plt.plot(x,y,'.',color='green')
plt.show()