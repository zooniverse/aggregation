#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import sys
import cPickle as pickle
import math
import matplotlib.pyplot as plt
import pymongo
import urllib
import matplotlib.cbook as cbook

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
#from divisiveDBSCAN import DivisiveDBSCAN
#from divisiveDBSCAN_multi import DivisiveDBSCAN
#from clusterCompare import metric,metric2
import logisticRegression

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

penguins = pickle.load(open(base_directory+"/Databases/penguins_vote__.pickle","rb"))

#does this cluster have a corresponding cluster in the gold standard data?
#ie. does this cluster represent an actual penguin?

# #user penguins for first image - with 5 images
# print len(penguins[5][0])
# #user data
# print penguins[5][0][0]
# #gold standard data
# #print penguins[5][0][1]
#
# #users who annotated the first "penguin" in the first image
# print penguins[5][0][0][0][1]
# #and their corresponds points
# print penguins[5][0][0][0][0]

lowest_cluster = float("inf")
highest_cluster = -float('inf')

#print gold_standard
#RESET
max_users = 20
y_values = []
for image in penguins[max_users]:
    #first - create a list of ALL users - so we can figure out who has annotated a "penguin" or hasn't
    for cluster in image[1]:

        X = np.mean(zip(*cluster[0])[0])
        Y = np.mean(zip(*cluster[0])[1])


        y_values.append(Y)

mid_point = np.mean(y_values)
plt.plot((0,10),(mid_point/100.,mid_point/100.))
low_dist = []
high_dist = []

overall_dist = []


overall_values = []
mid_cluster = []
for image_id in range(1):#len(penguins[max_users])):
    image = penguins[max_users][image_id]

    above_below = []

    for i in range(len(image[1])):
        closest_neighbours = []
        c_1 = image[1][i][0]
        X_1 = np.mean(zip(*c_1)[0])
        Y_1 = np.mean(zip(*c_1)[1])
        mid_cluster.append((1,X_1/100.,Y_1/100.))
        closest_dist = float("inf")
        closest = None
        #for j in range(i+1,len(low_clusters)):
        for j in range(len(image[1])):
            if i == j:
                continue
            c_2 = image[1][j][0]

            X_2 = np.mean(zip(*c_2)[0])
            Y_2 = np.mean(zip(*c_2)[1])

            if ((Y_1 < mid_point) and (Y_2 > mid_point)) or ((Y_1 > mid_point) and (Y_2 < mid_point)):
                continue

            dist = math.sqrt((X_1-X_2)**2+(Y_1-Y_2)**2)
            closest_neighbours.append(dist)

        #plt.plot((c_1[0],closest[0]),(c_1[1],closest[1]),color="blue")
        if closest_neighbours == []:
            #assert(len(low_clusters) == 1)
            pass
        else:
            closest_neighbours.sort()
            vv = np.mean(closest_neighbours[0:1])
            overall_values.append(vv)

high_values = []
low_values = []
overall_mean = np.mean(overall_values)
for c_1,value in zip(mid_cluster,overall_values):
    if value >= overall_mean:
        plt.plot((c_1[1]),(c_1[2]),"o",color= "red")
        above_below.append(1)
        high_values.append(value)
    else:
        plt.plot((c_1[1]),(c_1[2]),"o",color= "green")
        above_below.append(0)
        low_values.append(value)

print np.mean(high_values)
print np.mean(low_values)
# from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
# #clf = linear_model.LinearRegression()
#clf = SGDClassifier(loss="hinge", alpha=0.05, n_iter=1000, fit_intercept=True)
#clf.fit(np.array(mid_cluster), np.array(above_below))
#
#
xx = np.linspace(0, 10, 50)
yy = np.linspace(0, 6, 50)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)

min_p = float("inf")
max_p = -float("inf")
#print logisticRegression.cost_function((-20,-0.5,1),mid_cluster,above_below)

alpha = 0.2
theta = [-20,-0.5,1]

costs = []

for i in range(2000):
    #print i
    t1_temp = theta[0] - alpha*logisticRegression.partial_cost_function(theta,mid_cluster,above_below,0)
    t2_temp = theta[1] - alpha*logisticRegression.partial_cost_function(theta,mid_cluster,above_below,1)
    t3_temp = theta[2] - alpha*logisticRegression.partial_cost_function(theta,mid_cluster,above_below,2)

    theta = [t1_temp,t2_temp,t3_temp]
    costs.append(logisticRegression.cost_function(theta,mid_cluster,above_below))

#plt.plot(range(len(costs)),costs)
#plt.show()

for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    #p = clf.predict([x1, x2])

    p = logisticRegression.hypothesis(theta,(1,x1,x2))
    #if x2/600. < 0.5:
    #     print x2/600.
    #     print p

    #min_p = min(min_p,p)
    #max_p= max(min_p,p)
    #if math.fabs(0.5-p) < 0.01:
    #    print p

    Z[i, j] = p

#print min_p,max_p
#print clf.coef_
plt.contour(X1, X2, Z, [0.5], colors="blue")
# # #plt.plot(range(500),clf.predict(range(500)),color="blue")
#plt.xlim(0,10)
#plt.ylim(0,10)
plt.show()

#print mid_cluster
#print above_below