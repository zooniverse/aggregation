__author__ = 'ggdhines'
# import matplotlib
# matplotlib.use('WXAgg')
# from aggregation_api import AggregationAPI
import json
import os
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.cluster import DBSCAN
import numpy as np
import math

directory = "/home/ggdhines/Downloads/tmp/"
output_directory = "/home/ggdhines/cell/"
subjects = list(os.listdir(directory))
for f_count,fname in enumerate(subjects):
    if not fname.endswith(".jpg"):
        continue

    # fname = "1080284.jpg"

    gold_standard = json.load(open("/home/ggdhines/Downloads/tmp/gold_results.json","rb"))

    image_file = cbook.get_sample_data(directory+fname)
    image = plt.imread(image_file)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image)

    gray = rgb2gray(image)
    pts_x = []
    pts_y = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if gray[i][j] < 0.18:
                # plt.plot(i,j,color="blue")
                pts_x.append(j)
                pts_y.append(i)

    # plt.plot(pts_x,pts_y,".",color="blue")

    X = np.asarray(zip(pts_x,pts_y))
    db = DBSCAN(eps=10, min_samples=3).fit(X)
    labels = db.labels_

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    alg_pts = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            continue

        class_member_mask = (labels == k)
        if sum([1 for i in labels if i == k]) > 100:
            continue
        if sum([1 for i in labels if i == k]) < 6:
            continue

        xy = X[class_member_mask]
        x = np.median(xy[:, 0])
        y = np.median(xy[:, 1])
        alg_pts.append((x,y))
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor="green", markersize=2)
        # circle=plt.Circle((x,y),5,color='blue',fill=False)
        # ax.add_artist(circle)

    alg_to_gold = []#[] for _ in alg_pts]

    print gold_standard[fname]
    gold_pts = []
    for x,y,r in gold_standard[fname]:
        # plt.plot(x,y,"o",color="r")
        # circle=plt.Circle((x,y),5,color='r',fill=False)
        # ax.add_artist(circle)
        gold_pts.append((x,y))

    gold_to_alg = []#[] for _ in gold_pts]

    for j,(x_a,y_a) in enumerate(alg_pts):
        min_distance = float("inf")
        best_match = None

        for i,(x_g,y_g) in enumerate(gold_pts):
            dist = math.sqrt((x_a-x_g)**2+(y_a-y_g)**2)
            if (dist < min_distance) and (dist < 5):
                min_distance = dist
                best_match = i

        alg_to_gold.append(best_match)

    for i,(x_g,y_g) in enumerate(gold_pts):

        min_distance = float("inf")
        best_match = None

        for j,(x_a,y_a) in enumerate(alg_pts):
            dist = math.sqrt((x_a-x_g)**2+(y_a-y_g)**2)
            if (dist < min_distance) and (dist < 5):
                min_distance = dist
                best_match = j

        gold_to_alg.append(best_match)


    for i,(x,y) in enumerate(alg_pts):
        j = alg_to_gold[i]

        if (j is not None) and (gold_to_alg[j] == i):
            circle=plt.Circle((x,y),5,color='g',fill=False)
        else:
            circle=plt.Circle((x,y),5,color='y',fill=False)

        ax.add_artist(circle)

    for j,(x,y) in enumerate(gold_pts):
        i = gold_to_alg[j]

        if (i is None) or (alg_to_gold[i] != j):
            circle=plt.Circle((x,y),5,color='r',fill=False)
        ax.add_artist(circle)

    ax.set_axis_off()
    # ax.set_xticks()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')



    # plt.savefig(output_directory+fname,bbox_inches='tight', pad_inches=0)
    # plt.close()
    plt.show()