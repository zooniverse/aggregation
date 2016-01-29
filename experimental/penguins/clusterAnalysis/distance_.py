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

client = pymongo.MongoClient()
db = client['penguin_2014-10-22']
subject_collection = db["penguin_subjects"]

lowest_cluster = float("inf")
highest_cluster = -float('inf')

#print gold_standard
#RESET
max_users = 20
cluster_list = []
image = penguins[max_users][0]
for image in penguins[max_users]:
    #first - create a list of ALL users - so we can figure out who has annotated a "penguin" or hasn't
    zooniverse_id = image[0]
    for cluster in image[1]:

        X = np.mean(zip(*cluster[0])[0])
        Y = np.mean(zip(*cluster[0])[1])

        cluster_list.append((X,Y,cluster[1]))

    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    url = subject["location"]["standard"]
    object_id= str(subject["_id"])
    image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"
    if not(os.path.isfile(image_path)):
        urllib.urlretrieve(url, image_path)

    image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    image = plt.imread(image_file)
    fig, ax = plt.subplots()
    im = ax.imshow(image)

    to_plot = False

    for i in range(len(cluster_list)):
        for j in range(i+1,len(cluster_list)):
            users_1 = cluster_list[i][2]
            users_2 = cluster_list[j][2]

            overlap = len([u for u in users_1 if (u in users_2)])

            if overlap == 1:
                x_1,y_1 = cluster_list[i][0],cluster_list[i][1]
                x_2,y_2 = cluster_list[j][0],cluster_list[j][1]

                first_dist = math.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)

                min_dist = float("inf")
                #find the closest or next-closest cluster
                for k in range(len(cluster_list)):
                    if k in [i,j]:
                        continue


                    x_3,y_3 = cluster_list[k][0],cluster_list[k][1]

                    dist = math.sqrt((x_1-x_3)**2 + (y_1-y_3)**2)
                    min_dist = min(min_dist,dist)

                if first_dist< (min_dist):
                    to_plot = True
                    print first_dist,min_dist
                    plt.plot((x_1,x_2),(y_1,y_2))

    if to_plot:
        plt.show()
        plt.close()
    else:
        plt.close()

    print "===="