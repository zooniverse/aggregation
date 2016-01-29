#!/usr/bin/env python
import sys
import math

currentPhoto = None
counter = 0
currentPts = []
epsilon = 0.05

def fof(pts):
    clusters = []
    for p_x,p_y in pts:
        friendsWith = []
        for i,c in enumerate(clusters):
            for p2_x,p2_y in c:
                #is this point a friend
                if math.sqrt(math.fabs(p_x-p2_x)**2+ math.fabs(p_y-p2_y)**2) <= epsilon:
                    friendsWith.append(i)
                    break

        if friendsWith is []:
            clusters.append((p_x,p_y))
        else:
            #merge
            newCluster = []
            for clusterIndex in reversed(friendsWith):
                newCluster.extend(clusters.pop(clusterIndex))

            newCluster.append((p_x,p_y))
            clusters.append(newCluster)

    return clusters


# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    words = line.split("\t")

    photo = words[0]
    if photo != currentPhoto:
        if currentPhoto != None:
            print currentPhoto + "\t" + str(fof(currentPts))

        currentPhoto = photo
        currentPts = []

    p_x,p_y = words[1].split(",")
    currentPts.append((float(p_x),float(p_y)))


print currentPhoto + "\t" + str(fof(currentPts))