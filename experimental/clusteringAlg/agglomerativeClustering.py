#!/usr/bin/env python
__author__ = 'greg'
from sklearn.cluster import DBSCAN
import numpy as np
import math
from copy import deepcopy
from ete2 import Tree, TreeStyle



def toNewick(hierarchy,overallCounter = 1):
    assert(len(hierarchy) == 2)

    retval = "("

    if isinstance(hierarchy[0],list):
        d = averageLinkage(hierarchy[0][0],hierarchy[0][1])
        if d == float("inf"):
            d = "-1"
        else:
            d = str(int(d))
        retval += toNewick(hierarchy[0],overallCounter+1)
    else:
        retval += str(int(hierarchy[0][0])) + "-" + str(int(hierarchy[0][1]))

    retval += ","

    if isinstance(hierarchy[1],list):
        d = averageLinkage(hierarchy[1][0],hierarchy[1][1])
        if d == float("inf"):
            d = "-1"
        else:
            d = str(int(d))
        retval += toNewick(hierarchy[1],overallCounter+1)
    else:
        retval += str(int(hierarchy[1][0])) + "-" + str(int(hierarchy[1][1]))


    d = averageLinkage(hierarchy[0],hierarchy[1])
    if d == float("inf"):
        d = "-1"
    else:
        d = str(int(d))
    retval += ")"+d
    return retval

def dist(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

class CannotSplit(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return ""
samples_needed = 3

def flatten(l):
    if isinstance(l,list):
        if len(l) == 0:
            return []
        first, rest = l[0],l[1:]
        return flatten(first) + flatten(rest)
    else:
        return [l]


def getClusters(hierarchy):
    if isinstance(hierarchy,tuple):
        return []
    assert(len(hierarchy) == 2)

    clusters = []
    if averageLinkage(hierarchy[0],hierarchy[1]) == float("inf"):
        clusters.extend(getClusters(hierarchy[0]))
        clusters.extend(getClusters(hierarchy[1]))
    else:
        newCluster = flatten(hierarchy)
        if len(newCluster) >= 2:
            x,y,u = zip(*newCluster)
            clusters = [(np.mean(x),np.mean(y))]
        else:
            print "too small"

    return clusters


def averageLinkage(c1,c2):
    d = []
    flattened1 = flatten(c1)
    flattened2 = flatten(c2)

    for pt1 in flattened1:
        x1,y1,u1 = pt1
        for pt2 in flattened2:
            x2,y2,u2 = pt2
            if u1 == u2:
                d.append(float("inf"))
            else:
                d.append(math.sqrt((x2-x1)**2+(y2-y1)**2))

    return np.mean(d)

def mergeClusters(c1,c2):
    retval = c1[:]
    retval.extend(c2)

    return retval

def findClusterCenter(cluster):
    x,y,u = zip(*flatten(cluster))
    return np.mean(x),np.mean(y)


def agglomerativeClustering(pts):
    if pts == set([]):
        return []



    XYpts,user_ids = zip(*list(pts))

    X = np.array(XYpts)
    #use DBSCAN to create connectivity constraints
    #keep expanding until there is no more noise
    for first_epsilon in [25,50,100,200,300,400]:
        db = DBSCAN(eps=first_epsilon, min_samples=samples_needed).fit(X)

        if not(-1 in db.labels_):
            break



    labels = db.labels_
    end_clusters = []
    print max(labels)
    #do agglomerative clustering on each individual "sub" cluser
    for k in sorted(set(labels)):
        clusters = [(x,y,user) for (x,y), user, l in zip(XYpts,user_ids,labels) if l == k]
        #keep on merging as much as possible
        while len(clusters) > 1:
            minAvg = float("inf")
            bestChoice = None

            for i,c1 in enumerate(clusters):
                for j,c2 in enumerate(clusters[i+1:]):
                    clusterDist = averageLinkage(c1,c2)

                    if clusterDist <= minAvg:
                        minAvg = clusterDist
                        bestChoice = (i,i+1+j)

            i,j = bestChoice
            assert(j > i)
            c2 = clusters.pop(j)
            c1 = clusters.pop(i)
            clusters.append([c1,c2])

        end_clusters.append(deepcopy(clusters[0]))

    #now merge each of these "intermediate" clusters together
    #stop if the minAverage is infinity - at this point we are just merging pts from the same users
    while len(end_clusters) > 1:
        minAvg = float("inf")
        bestChoice = None
        for i,c1 in enumerate(end_clusters):
            for j,c2 in enumerate(end_clusters[i+1:]):
                clusterDist = averageLinkage(c1,c2)

                if clusterDist <= minAvg:
                    minAvg = clusterDist
                    bestChoice = (i,i+1+j)

        #if minAvg == float("inf"):
        #    break

        i,j = bestChoice
        assert(j > i)
        c2 = end_clusters.pop(j)
        c1 = end_clusters.pop(i)
        end_clusters.append([c1,c2])

    #print len(end_clusters[0])
    #print len(end_clusters[0][0])
    #print len(end_clusters[0][1])

    # newick = toNewick(end_clusters[0]) + ";"
    # print newick
    # t = Tree(newick)
    # ts = TreeStyle()
    # ts.show_leaf_name = True
    # #ts.show_branch_length = True
    # ts.show_branch_support = True
    # t.render("/home/greg/mytree.pdf", w=400, units="mm",tree_style=ts)
    # #t.show(tree_style=ts)
    return getClusters(end_clusters[0])
