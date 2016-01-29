#!/usr/bin/env python
__author__ = 'greg'
from sklearn.cluster import DBSCAN
import numpy as np
import math

def dist(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

class CannotSplit(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return ""
samples_needed = 3

def adaptiveDBSCAN(XYpts,user_ids):
    if XYpts == []:
        return []

    pts_in_each_cluster = []
    users_in_each_cluster = []
    cluster_centers = []

    #increase the epsilon until we don't have any nearby clusters corresponding to non-overlapping
    #sets of users
    X = np.array(XYpts)
    #for epsilon in [5,10,15,20,25,30]:
    for first_epsilon in [100,200,300,400]:
        db = DBSCAN(eps=first_epsilon, min_samples=samples_needed).fit(X)

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
    #print epsilon
    #print [c[1] for c in cluster_compare[:10]]
    centers_to_return = []
    assert not(needToMerge)


    #do we need to split any clusters?
    for cluster_index in range(len(cluster_centers)):
        #print "splitting"
        needToSplit = (sorted(users_in_each_cluster[cluster_index]) != sorted(list(set(users_in_each_cluster[cluster_index]))))
        if needToSplit:
            subcluster_centers = []
            stillToSplit = []
            X = np.array(pts_in_each_cluster[cluster_index])
            #for epsilon in [30,25,20,15,10,5,1,0.1,0.01]:
            for second_epsilon in range(200,1,-2):#[400,300,200,100,80,75,65,60,50,25,24,23,22,21,20,19,18,17,16,15,14,13,10,5,1]:
                db = DBSCAN(eps=second_epsilon, min_samples=samples_needed).fit(X)

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
                        stillToSplit = list(X[class_member_mask])
                        break

                    pts_in_cluster = list(X[class_member_mask])
                    xSet,ySet = zip(*pts_in_cluster)
                    subcluster_centers.append((np.mean(xSet),np.mean(ySet)))

                if not(needToSplit):
                    break


            if needToSplit:
                print "second is " + str(second_epsilon)
                print stillToSplit
                for i in range(len(stillToSplit)):
                    p1 = stillToSplit[i]
                    for j in range(len(stillToSplit[i+1:])):
                        p2 = stillToSplit[j+i+1]
                        print math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2),
                        #print (i,j+i+1),
                    print
                print X
                print users_in_each_cluster[cluster_index]
                raise CannotSplit(pts_in_each_cluster[cluster_index])
            centers_to_return.extend(subcluster_centers)

            #if needToSplit:
            #    print pts_in_each_cluster[cluster_index]
            #    print users_in_each_cluster[cluster_index]
            #else:

        else:
            centers_to_return.append(cluster_centers[cluster_index])



    return centers_to_return