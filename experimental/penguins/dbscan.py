#!/usr/bin/env python
import pymongo
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib
import matplotlib.cbook as cbook
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.patches import Ellipse
from copy import deepcopy
__author__ = 'greghines'

client = pymongo.MongoClient()
db = client['penguins']
collection = db["penguin_classifications"]

penguins = {}
adults = {}
chicks = {}
eggs = {}
count = {}
fNames = {}
pCount = {}

i = 0
pen = 0
total = 0
for r in collection.find():
    for a in r["annotations"]:
        if ('value' in a) and not(a["value"]  in ["penguin", "adult", "no", "yes", "finished", "unfinished", "cant_tell", "", "chick", "eggs", "other"]):

            zooniverseID = r["subjects"][0]["zooniverse_id"]
            if not(zooniverseID in adults):
                penguins[zooniverseID] = []
                adults[zooniverseID] = []
                chicks[zooniverseID] = []
                eggs[zooniverseID] = []
                count[zooniverseID] = 1
                url = r["subjects"][0]["location"]["standard"]
                fNames[zooniverseID] = url.split("/")[-1]
            else:
                count[zooniverseID] += 1

            penguins[zooniverseID].append(len(a["value"]))

            for index in a["value"]:
                point = a["value"][index]


                if point["value"] == "adult":
                    adults[zooniverseID].append((float(point["x"]),float(point["y"])))
                elif point["value"] == "chick":
                    chicks[zooniverseID].append((float(point["x"]),float(point["y"])))
                elif point["value"] == "eggs":
                    eggs[zooniverseID].append((float(point["x"]),float(point["y"])))
                else:
                    pass
                    #print point["value"]

                #penguins[zooniverseID].append((float(point["x"]),float(point["y"])))


overallCount = {2:0,3:0,4:0,5:0}

for zooniverseID in penguins:
    pCount[zooniverseID] = np.mean(penguins[zooniverseID])



#print sorted(pCount.items(),key = lambda x:x[1])
#assert(False)
print count["APZ00003i8"]
for zooniverseID,c in count.items():
    if c >= 3:


        #overallCount[c] += 1
        #if zooniverseID in ["APZ00004er","APZ00004er","APZ00003h1"]: #!= "APZ00003lc":
        #    continue
        print str(zooniverseID) + "," + str(c)

        #print "/home/greg/Databases/penguins/images"+fNames[zooniverseID]

        #print zooniverseID,pCount[zooniverseID]

        #if zooniverseID in ["APZ00003lc","APZ00002ea","APZ00003l4"]:
        #    continue
        #print (zooniverseID,c)
        #do we already have this file?
        if not(os.path.isfile("/home/greg/Databases/penguins/images/"+fNames[zooniverseID])):
            urllib.urlretrieve ("http://demo.zooniverse.org/penguins/subjects/standard/"+fNames[zooniverseID], "/home/greg/Databases/penguins/images/"+fNames[zooniverseID])

        image_file = cbook.get_sample_data("/home/greg/Databases/penguins/images/"+fNames[zooniverseID])
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)

        fOut = open("/home/greg/Databases/penguins/dbscan/"+fNames[zooniverseID][:-4]+".csv","wb")
        fOut.write("penguinType,xCoord,yCoord\n")

        for colour,penType,data in [("green","adult",adults),("blue","chick",chicks),("red","egg",eggs)]:
            X = np.array(data[zooniverseID])
            db = DBSCAN(eps=20, min_samples=2).fit(X)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_



            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            #print('Estimated number of clusters: %d' % n_clusters_)

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))




            for k in unique_labels:
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                if k == -1:
                    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k',markeredgecolor='k', markersize=6)
                else:
                    xSet,ySet = zip(*list(X[class_member_mask]))
                    x = np.mean(xSet)
                    y = np.mean(ySet)
                    plt.plot(x, y, 'o', markerfacecolor=colour,markeredgecolor='k', markersize=6)
                    fOut.write(penType+","+str(x)+","+str(y)+"\n")

        fOut.close()
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = 'k'
        #
        #     class_member_mask = (labels == k)
        #
        #     xy = X[class_member_mask & core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=14)
        #
        #     xy = X[class_member_mask & ~core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=6)
        #
        plt.title('Number of users: %d' % (c))
        plt.savefig("/home/greg/Databases/penguins/dbscan/"+fNames[zooniverseID])
        plt.close()
        #plt.show()
        #break

print overallCount