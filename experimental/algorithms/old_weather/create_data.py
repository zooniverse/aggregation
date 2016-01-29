__author__ = 'greg'
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import cPickle as pickle

# img = cv2.imread("/home/ggdhines/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
# print type(img)
# assert False

with open('old_weather.json') as data_file:
    data = json.load(data_file)

scale_x = 1.455
scale_y = 1.45
x_offset = 20
y_offset = 2

# image_file = cbook.get_sample_data("/home/ggdhines/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
image = plt.imread("/home/ggdhines/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")

base = [198, 188, 153]

range_x = 0
range_y = 0

def med_x(pts):
    x,y = zip(*pts)
    return np.median(x)

data_c = 0

cluster_fig = None

for box in data["classifications"][0]["annotations"]:
    if "type" in box.keys():
        continue

    # fig = plt.figure()
    # axes = fig.add_subplot(1, 1, 1)
    # im = axes.imshow(image)
    x = int((box["x"]-x_offset)/scale_x)
    y = int((box["y"]-y_offset)/scale_y)

    x2 = int((box["x"] + box["width"]-x_offset)/scale_x)
    y2 = int((box["y"] + box["height"]-y_offset)/scale_y)
    #
    # # if ("type" in box.keys()) or (y < -1200):
    # #     continue
    #
    # plt.plot([x,x2,x2,x,x],[y,y,y2,y2,y])
    #
    # plt.show()

    m = []

    for y_i in range(y,y2+1):
    # for x_i in range(x,x2+1):
        m_t = []
        # for y_i in range(y,y2+1):
        for x_i in range(x,x2+1):
            p = image[y_i][x_i]
            m_t.append(p)

        m.append(np.asarray(m_t))
    # print np.asarray(m)
    # assert False
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    m = np.asarray(m)
    axes.imshow(m)

    plt.ion()
    plt.show()

    number = raw_input("Enter the text: ")
    number_list = [n for n in number]
    print number_list




    if number == "":
        break

    # now create the clusters
    pts = []
    # values are flipped
    min_y = 0
    max_y = -float("inf")

    max_x = 0
    min_x = float("inf")

    for y_i in range(y,y2+1):
    # for x_i in range(x,x2+1):
        # for y_i in range(y,y2+1):
        for x_i in range(x,x2+1):
            p = image[y_i][x_i]
            if math.sqrt(sum([(a-b)**2 for (a,b) in zip(base,p)])) > 45:
                pts.append((x_i,-y_i))
                # plt.plot(x_i,-y_i,'o',color="black")
                min_y = min(-y_i,min_y)
                max_y = max(-y_i,max_y)

                min_x = min(x_i,min_x)
                max_x = max(x_i,max_x)

    if len([y for (x,y) in pts if y == min_y]) > 4:
        print "****"
        for i in range(len(pts)-1,-1,-1):
            if pts[i][1] == min_y:
                del pts[i]

        # reset min y
        min_y = min([y for (x,y) in pts])

    # print max_y-min_y
    # print len([y for (x,y) in pts if y == min_y])
    # print len([y for (x,y) in pts if y == max_y])
    # print len([y for (x,y) in pts if x == min_x])
    # print len([y for (x,y) in pts if x == max_x])

    # plt.show()








    print [min_x,max_x]
    print [min_y,max_y]
    print "===="

    # repeat until we have removed all border cases - since we don't know which order we are going in
    border_case = True
    while border_case:
        pts = np.asarray(pts)
        db = DBSCAN(eps=1.5, min_samples=2).fit(pts)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        border_case = False

        # each time we iterate through, start with a fresh graph
        cluster_fig = plt.figure()
        axes = cluster_fig.add_subplot(1, 1, 1)


        # in case we need to repeat
        new_pts = []
        clusters = []
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                continue

            class_member_mask = (labels == k)

            xy = pts[class_member_mask]
            x,y = zip(*xy)

            x_r = max(x)-min(x)
            y_r = max(y)-min(y)

            if (max(x) == min(x)) and (max(x) in [min_x,max_x]):
                print "border case"
                border_case = True
                continue
            if (max(y) == min(y)) and (max(y) in [min_y,max_y]):
                print "border case"
                border_case = True
                continue

            new_pts.extend(xy)

            print min(x),max(x)
            print min(y),max(y)
            print

            # range_x = max(range_x,x_r)
            # range_y = max(range_y,y_r)

            clusters.append(xy)
            axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)

        if border_case:
            pts = new_pts
            plt.close(cluster_fig)

            x,y = zip(*pts)
            max_x = max(x)
            min_x = min(x)
            max_y = max(y)
            min_y = min(y)



    # plt.show()
    if len(clusters) == len(number_list):
        print "match!"
        clusters.sort(key = lambda c:med_x(c))

        for n,c in zip(number_list,clusters):
            pickle.dump((c,n),open("/home/ggdhines/Dropbox/nn_cases/"+str(data_c)+".pic","wb"))
            data_c += 1

    else:
        t = raw_input("enter anything")
    plt.close(fig)
    plt.close(cluster_fig)


        # for c in clusters:
        #     x,y = zip(*c)
        #     print np.median(x)
        # print

