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

        lowest_cluster = min(lowest_cluster,Y)
        highest_cluster = max(highest_cluster,Y)

        y_values.append(Y)

mid_point = (lowest_cluster+highest_cluster)/2.
mid_point = np.mean(y_values)
low_dist = []
high_dist = []

overall_dist = []

client = pymongo.MongoClient()
db = client['penguin_2014-10-22']
subject_collection = db["penguin_subjects"]



for image in penguins[max_users]:
    low_clusters = []
    high_clusters = []

    subject = subject_collection.find_one({"zooniverse_id":image[0]})
    url = subject["location"]["standard"]
    object_id= str(subject["_id"])
    image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"
    if not(os.path.isfile(image_path)):
        urllib.urlretrieve(url, image_path)

    image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    penguin_image = plt.imread(image_file)
    fig, ax = plt.subplots()
    im = ax.imshow(penguin_image)
    #plt.plot((0,1000),(mid_point,mid_point))
    #plt.show()

    exception_found = False

    for cluster in image[1]:
        X = np.mean(zip(*cluster[0])[0])
        Y = np.mean(zip(*cluster[0])[1])

        #REMEMBER - image is flipped!! (no idea why)
        if Y > mid_point:
            low_clusters.append(((X,Y),cluster[1]))
        else:
            high_clusters.append(((X,Y),cluster[1]))

    for i in range(len(low_clusters)):
        closest_neighbours = []
        c_1 = low_clusters[i][0]
        closest_dist = float("inf")
        closest = None
        #for j in range(i+1,len(low_clusters)):
        for j in range(len(low_clusters)):
            if i == j:
                continue
            c_2 = low_clusters[j][0]

            users_1 = low_clusters[i][1]
            users_2 = low_clusters[j][1]

            overlap = len([u for u in users_1 if (u in users_2)])

            dist = math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2)

            if True: #overlap == 1:

                #low_dist.append(dist)
                #overall_dist.append(math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2))
                closest_neighbours.append((dist,c_2,overlap))

        #plt.plot((c_1[0],closest[0]),(c_1[1],closest[1]),color="blue")
        if closest_neighbours == []:
            assert(len(low_clusters) == 1)
        else:
            closest_neighbours.sort(key = lambda x:x[0])
            vv = np.mean(zip(*closest_neighbours[0:2])[0])
            low_dist.append(vv)

            if vv < 12:
                closest_cluster = closest_neighbours[0][1]
                plt.plot((c_1[0],closest_cluster[0]),(c_1[1],closest_cluster[1]))
                print "==== " + str(closest_neighbours[0][2])
                exception_found = True

    for i in range(len(high_clusters)):
        closest_neighbours = []
        #for j in range(i+1,len(high_clusters)):
        for j in range(len(high_clusters)):
            if i == j:
                continue

            c_1 = high_clusters[i][0]
            c_2 = high_clusters[j][0]

            users_1 = high_clusters[i][1]
            users_2 = high_clusters[j][1]

            overlap = len([u for u in users_1 if (u in users_2)])

            if True: #overlap == 1:
                dist = math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2)
                #high_dist.append(dist)
                #overall_dist.append(math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2))
                closest_neighbours.append(dist)

        if closest_neighbours == []:
            assert(len(high_clusters))
        else:
            closest_neighbours.sort()
            high_dist.append(np.mean(closest_neighbours[0:2]))

    if not exception_found:
        plt.close()
    else:
        plt.show()
        #plt.close()

print np.mean(low_dist)
print np.median(low_dist)

n, bins, patches  =plt.hist(low_dist, 1000, normed=1,histtype='step', cumulative=True,color="green")
print n[3],bins[3]
print "====="
print np.mean(high_dist)
print np.median(high_dist)
n, bins, patches  =plt.hist(high_dist, 1000, normed=1,histtype='step', cumulative=True,color="blue")
print n[3],bins[3]
plt.xlim(0,200)
plt.ylim(0,0.4)
#plt.show()