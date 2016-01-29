#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os
import math
import sys

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from clusterCompare import cluster_compare

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

#have as a list not a tuple since we need the index
gold_standard = []

#create the gold standard data
max_users = 20
image_index = 0
#first - create a list of ALL users - so we can figure out who has annotated a "penguin" or hasn't
for cluster in penguins[max_users][image_index][1]:
    gold_standard.append(cluster[0])

#print gold_standard
#RESET
max_users = 5
#first - create a list of ALL users - so we can figure out who has annotated a "penguin" or hasn't
user_set = []

for cluster in penguins[max_users][image_index][1]:
    users = cluster[1]
    for u in users:
        if not(u in user_set):
            user_set.append(u)

#now actually figure out how has annotated a penguin or hasn't
f = open(base_directory+"/Databases/penguins_ibcc.in",'wb')
f.write("a,b,c\n")

for cluster_index,cluster in enumerate(penguins[max_users][image_index][1]):
    users = cluster[1]
    for user_index,u_ in enumerate(user_set):
        if u_ in users:
            #print (user_index,penguin_index,1)
            f.write(str(user_index)+","+str(cluster_index) + ",1\n")
        else:
            #iprint (user_index,penguin_index,0)
            f.write(str(user_index)+","+str(cluster_index) + ",0\n")

    print cluster_index

#now do the analysis
# for image_index in range(len(penguins[5])):
#
#
#     user_penguins = penguins[5][0]
#     gold_penguins = penguins[5][1]
#
#     print len(user_penguins)
#     print len(user_penguins[0])
#
#     print "==="
#     print len(user_penguins)
#     for upen in user_penguins:
#         print upen
#         print cluster_compare(gold_penguins,[upen,])
#
#     break


with open(base_directory+"/Databases/penguins_ibcc_config.py",'wb') as f:
    f.write("import numpy as np\nscores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("inputFile = '"+base_directory+"/Databases/penguins_ibcc.in'\n")
    f.write("outputFile =  '"+base_directory+"/Databases/penguins_ibcc.out'\n")
    f.write("confMatFile = '"+base_directory+"/Databases/penguins_ibcc.mat'\n")
    f.write("nu0 = np.array([10.0,90.0])\n")
    f.write("alpha0 = np.array([[2,1], [1, 9]])\n")

if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
else:
    sys.path.append("/home/greg/github/pyIBCC/python")
import ibcc

try:
    os.remove(base_directory+"/Databases/penguins_ibcc.out")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/penguins_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/penguins_ibcc.in.dat")
except OSError:
    pass

ibcc.runIbcc(base_directory+"/Databases/penguins_ibcc_config.py")

print "done that"

total = 0
true_positives = []
false_positives = []
with open(base_directory+"/Databases/penguins_ibcc.out",'rb') as f:
    for l in f.readlines():
        penguin_index, neg_prob,pos_prob = l.split(" ")

        penguin = penguins[max_users][image_index][1][int(float(penguin_index))][0]

        #is this penguin "real" ie. is in the gold standard?
        if cluster_compare(gold_standard,[penguin,]) == []:
            #yes - penguin is real
            true_positives.append(float(pos_prob))
        else:
            #penguin is fake
            false_positives.append(float(pos_prob))

print min(sorted(true_positives)[2:])
print max(false_positives)
print len(false_positives)

X = []
Y = []
for p in np.arange(0,1.01,0.001):
    X.append(len([f for f in false_positives if f >= p])/float(len(false_positives)))
    Y.append(len([t for t in true_positives if t >= p])/float(37.))

plt.plot(X,Y,'-o')
plt.plot((0,1),(0,1),'-')
plt.show()