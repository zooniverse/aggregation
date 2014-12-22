#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import cPickle as pickle
import bisect
import csv
import matplotlib.pyplot as plt
import random
import math
import urllib
import matplotlib.cbook as cbook
from scipy.stats.stats import pearsonr
from scipy.stats import beta

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/classifier")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
    sys.path.append("/home/greg/github/reduction/experimental/classifier")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from divisiveKmeans import DivisiveKmeans
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()

client = pymongo.MongoClient()
db = client['penguin_2014-10-22']
collection = db["penguin_classifications"]
collection2 = db["penguin_subjects"]

X = []
Y = []
Z = []
W = []

XYZW = []

count = 0
totalPenguins = 0
#SALIa/SALIa2013a
for subject_index,subject in enumerate(collection2.find({"metadata.path":{"$regex" : ".*BAILa2014a.*"}})):
    path = subject["metadata"]["path"]



    #print path

    if not("BAILa2014a" in path):
        continue

    if count == 100:
        break

    print count

    count += 1

    user_markings = []
    user_ips = []

    zooniverse_id = subject["zooniverse_id"]
    print zooniverse_id
    for r in collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
        ip = r["user_ip"]
        n = 0
        xy_list = []
        try:
            if isinstance(r["annotations"][1]["value"],dict):
                for marking in r["annotations"][1]["value"].values():
                    if marking["value"] in ["adult","chick"]:
                        x,y = (float(marking["x"]),float(marking["y"]))
                        user_markings.append((x,y))
                        user_ips.append(ip)
        except KeyError:
            print r["annotations"]

    user_identified_condors,clusters,users = DivisiveKmeans(1).fit2(user_markings,user_ips,debug=True)

    totalPenguins += len(user_identified_condors)

    if user_identified_condors == []:
        continue

    for cluster_index in range(len(user_identified_condors)):
    #for x1,y1 in user_identified_condors:
        x1,y1 = user_identified_condors[cluster_index]
        if (x1 < 0) or (y1 < 0):
            continue
        u1 = users[cluster_index]
        #find the next closest point
        dist = [(math.sqrt((x1-x2)**2+(y1-y2)**2),(x2,y2)) for (x2,y2) in user_identified_condors]
        overlap = [len([u for u in u2 if u in u1]) for u2 in users]
        overlap2 = [[u for u in u2 if u in u1] for u2 in users]
        merged = zip(dist,overlap,overlap2)
        merged.sort(key = lambda x:x[0][0])
        next_closest = merged[1]
        X.append(-y1)
        Y.append(next_closest[0][0])
        #Z.append(next_closest[1])
        #W.append((zooniverse_id,user_identified_condors[cluster_index]))

        #y value, distance to closest point, over lap
        XYZW.append(((x1,y1),next_closest[0][0],next_closest[0][1],next_closest[1],zooniverse_id,next_closest[2]))



XYZW.sort(key=lambda x: x[0][1])


XY = sorted(zip(X,Y),key=lambda x:x[0])
lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]


#print pearsonr(X,Y)
#plt.plot(Y,X,'.')
#plt.show()

less_05 = 0
less_1 = 0
between_1_2 = 0
print totalPenguins

for jj,piece in enumerate(lol(XYZW,500)):
    #print piece
    #sort by distance
    piece.sort(key = lambda x:x[1])
    #assert False
    print "===---==="
    #print len([1 for pt1,dist,pt2,overlap,zooniverse_id in piece if overlap == 1])

    #compare S against all data

    temp = [piece[i][1] for i in range(len(piece))]# if piece[i][3] == 1]
    sampleMean = np.mean(temp)
    sampleVariance = np.var(temp,ddof=1)
    c = max(temp)
    a = 0#min(temp)
    x_bar = (sampleMean-a)/(c-a)
    v_bar = sampleVariance/((c-a)**2)

    alpha_bar = x_bar*(x_bar*(1-x_bar)/v_bar - 1)
    beta_bar = (1-x_bar)*(x_bar*(1-x_bar)/v_bar - 1)

    temp = [piece[i][1] for i in range(len(piece)) if piece[i][3] == 1]
    normalized_temp = [(t-a)/(c-a) for t in temp]
    temp2 = [(piece[i][1],piece[i][5]) for i in range(len(piece))]# if piece[i][3] == 1]
    temp2.sort(key = lambda x:x[0])
    #print zip(*temp2)[1][0:10]
    percentiles = [beta.cdf(x,alpha_bar,beta_bar) for x in normalized_temp]

    x_bar = np.mean(percentiles)
    v_bar = np.var(percentiles,ddof=1)

    alpha_bar = x_bar*(x_bar*(1-x_bar)/v_bar - 1)
    beta_bar = (1-x_bar)*(x_bar*(1-x_bar)/v_bar - 1)

    print percentiles[0:10]
    #print v_bar < x_bar*(1-x_bar)
    #print len(percentiles)
    #print beta.cdf(percentiles[0],alpha_bar,beta_bar)

    temp = [piece[i][1] for i in range(len(piece)) if piece[i][3] == 1]
    sampleMean = np.mean(temp)
    sampleVariance = np.var(temp,ddof=1)
    c = max(temp)
    a = 0#min(temp)
    x_bar = (sampleMean-a)/(c-a)
    v_bar = sampleVariance/((c-a)**2)
    alpha_bar = x_bar*(x_bar*(1-x_bar)/v_bar - 1)
    beta_bar = (1-x_bar)*(x_bar*(1-x_bar)/v_bar - 1)
    normalized_temp = [(t-a)/(c-a) for t in temp]
    percentiles = [beta.cdf(x,alpha_bar,beta_bar) for x in normalized_temp]
    print percentiles[0:10]


    n, bins, patches = plt.hist(percentiles,bins=20,normed=1,histtype='step',cumulative=True)
    Y = [beta.cdf(x,alpha_bar,beta_bar) for x in bins]
    plt.plot(bins,Y)
    #print bins
    plt.show()
    #assert False

    #print percentiles[0:10]

    for p in percentiles:
        if p < 0.005:
            less_05 += 1
        elif p < 0.01:
            less_1 += 1
        elif p < 0.02:
            between_1_2 += 1
        else:
            break

    #continue

    # print v_bar < x_bar*(1-x_bar)
    # #print temp
    # #print sorted(temp)
    # print len(normalized_temp)
    # tempX = range(len(temp))
    # regression = np.polyfit(tempX, temp, 3)
    # p = np.poly1d(regression)
    #
    # plt.plot(tempX,temp,"o-")
    # tempZ = [p(x) for x in tempX]
    # plt.plot(tempX,tempZ,"-")
    # plt.show()
    #
    # plt.plot(tempX,temp,"o-")
    # #tempZ = [p(x) for x in tempX[:100]]
    # plt.plot(tempX,tempZ,"-")
    # plt.xlim((0,50))
    # plt.ylim((0,50))
    # plt.show()
    #
    #
    # plt.hist(temp)
    # plt.show()
    # if jj == 5:
    #     break

    temp = [i for i in range(len(piece)) if piece[i][3] == 1]
    pt1,dist,pt2,overlap,zooniverse_id,ss = piece[temp[0]]

    #print i/float(len(piece)),dist

    r = collection2.find_one({"zooniverse_id":zooniverse_id})
    path = r["metadata"]["path"]
    object_id= str(r["_id"])
    url = r["location"]["standard"]
    image_path = base_directory+"/Databases/penguins/images/"+object_id+".JPG"

    if not(os.path.isfile(image_path)):
        urllib.urlretrieve(url, image_path)

    image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
    image = plt.imread(image_file)
    fig, ax = plt.subplots()
    im = ax.imshow(image)

    plt.plot((pt1[0],pt2[0]),(pt1[1],pt2[1]),"-")
    plt.show()


    # print len(piece)
    # X,Y,Z = zip(*piece)
    #
    # meanX = np.mean(X)
    # meanY = np.mean(Y)
    # varianceY = np.var(Y,ddof=1)
    # print varianceY
    # #print zip(*sorted(zip(Y,Z),key = lambda x:x[0]))[1]
    # #continue
    # #print varianceY < (meanY*(1-meanY))
    #
    # print [i/float(len(Z)) for i in range(len(Z)) if Z[i] == 1]
    #
    # #plt.hist(Y,15)
    # #plt.show()
    # maxX = max(X)
    # plt.plot((maxX,maxX),(0,250),color="red")

#plt.show()

print less_05
print less_1
print between_1_2