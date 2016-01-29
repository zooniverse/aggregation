#!/usr/bin/env python
import pymongo
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pylab import figure, show, rand
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.patches import Ellipse
from copy import deepcopy
__author__ = 'greghines'

client = pymongo.MongoClient()
db = client['milky_way']
collection = db["milky_way_classifications"]

goldStandardIDs = [u'AMW0000tf7', u'AMW0000rvj', u'AMW0000qwf', u'AMW0000fu3', u'AMW0000ieg', u'AMW000079z', u'AMW00007wk', u'AMW0000p96', u'AMW0000puo', u'AMW0000tz7', u'AMW0000k6s', u'AMW0000t55', u'AMW0000jec', u'AMW0000u7h', u'AMW0000u6e', u'AMW0000poo', u'AMW0000nqo', u'AMW0000guf', u'AMW0000uc5', u'AMW0000t4p', u'AMW0000po2', u'AMW0000px9', u'AMW0000v8d', u'AMW0000p8e', u'AMW0000pu1', u'AMW0000v1f', u'AMW0000vah']
goldStandardIDs = ["AMW0000t4p"]

#goldStandardData = []
goldClassification = collection.find_one({"subjects.zooniverse_id":"AMW0000t4p","user_name":"ttfnrob"})
goldX = float(goldClassification["annotations"][0]["center"][0])
goldY = float(goldClassification["annotations"][0]["center"][1])
goldHeight = float(goldClassification["annotations"][0]["rx"])
goldWeight = float(goldClassification["annotations"][0]["ry"])
goldRotation = float(goldClassification["annotations"][0]["angle"])

def ellipseIntersection(ellipse1,ellipse2):
    (x1, y1, h1, w1, r1) = ellipse1
    (x2, y2, h2, w2, r2) = ellipse2

    delta1 = max(h1,w1)
    delta2 = max(h2,w2)

    lowerX = min(x1 - delta1,x2 - delta2)
    lowerY = min(y1 - delta1,y2 - delta2)
    upperX = max(x1 + delta1,x2 + delta2)
    upperX = max(x1 + delta1,x2 + delta2)

def avgEllipse(circles):
    avgX = sum([x for (x,y,h,w,r) in circles])/float(len(circles))
    avgY = sum([y for (x,y,h,w,r) in circles])/float(len(circles))
    avgHeight = sum([h for (x,y,h,w,r) in circles])/float(len(circles))
    avgWidth = sum([w for (x,y,h,w,r) in circles])/float(len(circles))
    avgRotation = sum([r for (x,y,h,w,r) in circles])/float(len(circles))

    print ((avgX,avgY),avgHeight,avgWidth,avgRotation)

    return Ellipse((avgX,avgY),avgHeight,avgWidth,avgRotation)

for subjectID in goldStandardIDs:
    dataPts = []
    fullDataPts = []
    for classification in collection.find({"subjects.zooniverse_id":subjectID}):
        for annotation in classification["annotations"]:
            try:
                if True:#annotation["name"] == "cluster":
                    centerX,centerY = annotation["center"]
                    height = float(annotation["rx"])
                    width = float(annotation["rx"])
                    rotation = float(annotation["angle"])
                    dataPts.append([float(centerX),float(centerY)])
                    fullDataPts.append([float(centerX),float(centerY),height,width,rotation])
            except KeyError:
                break


    dataPts = np.array(dataPts)
    fullDataPts = np.array(fullDataPts)
    db = DBSCAN(eps=20, min_samples=25).fit(dataPts)
    labels = db.labels_

    unique_labels = set(labels)
    clusteredData = [[] for i in range(len(unique_labels)-1)]
    for i,label in enumerate(db.labels_):
        if label != -1:
            clusteredData[label].append(fullDataPts[i][:])

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)



    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        print k
        if k == 0:
            continue

        class_member_mask = (labels == k)

        xy = fullDataPts[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

        xy = fullDataPts[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    ax.plot(goldX,goldY,'rs-.')
    ax.add_artist(Ellipse((goldX,goldY),goldWeight, goldHeight, goldRotation))
    e = avgEllipse(clusteredData[0])
    ax.add_artist(e)
    e.set_facecolor(rand(3))
    #print avgEllipse(clusteredData[0])
    #ax.title('Estimated number of clusters: %d' % n_clusters_)
    show()


    break
