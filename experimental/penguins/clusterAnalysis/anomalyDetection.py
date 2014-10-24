#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os
import math
import sys
import urllib
import matplotlib.cbook as cbook

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

client = pymongo.MongoClient()
db = client['penguin_2014-10-12']
subject_collection = db["penguin_subjects"]

#print gold_standard
#RESET
max_users = 20
image_index = 0
for image_index in range(len(penguins[20])):
    #first - create a list of ALL users - so we can figure out who has annotated a "penguin" or hasn't
    user_set = []


    cluster_dict = {}
    #image = penguins[max_users][image_index]
    penguin_clusters =  penguins[max_users][image_index][1]
    zooniverse_id = penguins[max_users][image_index][0]


    for penguin_index in range(len(penguin_clusters)):
        users = penguin_clusters[penguin_index][1]
        cluster = penguin_clusters[penguin_index][0]
        center_x = np.mean(zip(*cluster)[0])
        center_y = np.mean(zip(*cluster)[1])

        cluster_dict[(center_x,center_y)] = users

    cluster_list = cluster_dict.keys()
    relations = []
    for i in range(len(cluster_list)-1):
        for j in range(i+1,len(cluster_list)):
            c_1 = cluster_list[i]
            c_2 = cluster_list[j]

            users_1 = cluster_dict[c_1]
            users_2 = cluster_dict[c_2]

            dist = math.sqrt((c_1[0]-c_2[0])**2+(c_1[1]-c_2[1])**2)
            overlap = len([u for u in users_1 if (u in users_2)])
            relations.append((dist,overlap,(i,j)))

    relations.sort(key = lambda x:x[0])

    user_relations = zip(*relations)[1]
    cluster_tuples = zip(*relations)[2]
    try:

        closest_single_connection = user_relations.index(1)

        if closest_single_connection > 0:
            print "no error"
            continue
        print relations[0:10]
        #we have an error
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

        for ii in range(min(len(user_relations),1)):
            if user_relations[ii] == 1:
                print ii
                i,j = cluster_tuples[ii]
                c_1 = cluster_list[i]
                c_2 = cluster_list[j]

                #X,Y = zip(*cluster_list)
                #plt.plot(X,Y,'o')

                X,Y = zip(*(c_1,c_2))
                plt.plot(X,Y,'-',color="blue")

        plt.show()

    except ValueError:
        print "**"