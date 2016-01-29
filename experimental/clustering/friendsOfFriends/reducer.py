#!/usr/bin/env python
import sys
import math

epsilon = 0.5

currentClusters = []
closed = []
maxClusterDist = []
minDist = None
maxDist = None
currDist = None

# input comes from STDIN (standard input)
for line in sys.stdin:
    newDist,pt = line.split("\t")
    xPt,yPt = pt.split(",")

    newDist = float(newDist)
    xPt = float(xPt)
    yPt = float(yPt)

    if minDist == None:
        minDist = newDist

    maxDist = newDist

    #check to see if this point is a "friend" of any of the pre-existing clusters
    #basically brute force it so check to see if any points are within epsilon of this new point
    #may have to merge clusters

    friendsWith = []
    for clusterIndex, cluster in enumerate(currentClusters):
        if closed[clusterIndex] == True:
            continue

        for existingPt in cluster:
            if math.sqrt(math.abs(xPt-existingPt[0])**2 + math.abs(yPt-existingPt[1])**2) <= epsilon:
                friendsWith.append(clusterIndex)
                #only need to fine one point that we are "friends" with
                break

    #if we found no friends, treat this as a cluster of one
    if friendsWith == []:
        currentClusters.append((xPt,yPt))
        maxClusterDist.append(newDist)
        closed.append(False)
    else:
        #merge
        newCluster = []
        for clusterIndex in friendsWith.reverse():
            newCluster.extend(currentClusters.pop(clusterIndex))
            closed.pop(clusterIndex)

        newCluster.append((xPt,yPt))
        newCluster.sort(key = lambda pt: math.sqrt(pt[0]**2+pt[1]**2))
        currentClusters.append(newCluster)
        closed.append(False)

    #check to see if any of the clusters can be "closed"
    for clusterIndex in range(len(currentClusters))
        pt = currentClusters[clusterIndex][-1]
        maxDist = math.sqrt(pt[0]**2+pt[1]**2)

        if (newDist - maxDist) > (math.sqrt(2)*epsilon):
            closed[clusterIndex] = True

for

