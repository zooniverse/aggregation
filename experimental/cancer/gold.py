#!/usr/bin/env python
__author__ = 'greghines'
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import re

goldX_1 = [331807.4,   3556826.0,  20072830.4,  20366014.0,  21147836.6,  24079671.8,  24666038.8,  31506987.4,  32288810.0,  38543391.6,  71282217.0, 140180342.0]
goldY_1 = [0.4605457, 0.5014045, 0.4975740, 0.3315852, 0.4860825, 0.4937435, 0.3456304, 0.3443535, 0.4975740, 0.4988508, 0.4975740, 0.4975740]

client = pymongo.MongoClient()
db = client['ouroboros_sandbox_2015-01-05']
classification_collection = db["cancer_gene_runner_classifications"]

user_points = []

for classification in classification_collection.find({"annotations.0":{"fileName" : "MB-3031Chrom9.txt"}}):
    pathString = classification["annotations"][1]["pathData"]
    pts = pathString.split(",")
    P = []
    for i in range(len(pts)/2):
        x = pts[2*i]
        y = pts[2*i+1]

        colonIndex = x.index(":")
        x = float(x[colonIndex+1:])

        colonIndex = y.index(":")
        bracketIndex = y.index("}")
        y = float(y[colonIndex+1:bracketIndex])

        P.append((x,y))

    print P
    user_points.append(P[:])


for jj in range(10):
    gold_x = goldX_1[jj]
    gold_y = goldY_1[jj]

    p = []
    for classification in user_points:
        #find the largest X point smaller than x
        tempY = None
        print classification
        for pts in classification:
            print pts
            x,y = pts

            if x < gold_x:
                tempY = y
            else:
                break

        assert tempY is not None
        p.append(tempY)

    print np.mean(p)
    print np.median(p)
    plt.hist(p,normed=1, histtype='step')
    plt.show()
