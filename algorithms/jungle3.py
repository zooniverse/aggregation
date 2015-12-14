__author__ = 'ggdhines'
import matplotlib
matplotlib.use('WXAgg')
import aggregation_api
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

ref = [[234,218,209],]


def analyze(f_name,display=False):
    image = cv2.imread(f_name)
    x_lim,y_lim,_ = image.shape
    a = image.reshape(x_lim*y_lim,3)

    dist = distance.cdist(a,ref).reshape(x_lim,y_lim)
    y_pts,x_pts = np.where(dist>50)

    pts = zip(x_pts,y_pts)
    pts.sort(key = lambda p:p[0])
    print "here"

    current_x = pts[0][0]

    # clusters = [[pts[0][1],],]

    to_plot = {}

    to_cluster_y = []

    for (x,y) in pts:
        if x == current_x:
            to_cluster_y.append(y)
        else:
            to_cluster_y.sort()
            to_plot[current_x] = [[to_cluster_y[0]],]

            for p in to_cluster_y[1:]:
                # print to_plot
                if (p - to_plot[current_x][-1][-1]) > 2:
                    to_plot[current_x].append([p])
                else:
                    to_plot[current_x][-1].append(p)

            to_cluster_y = []
            current_x = x

    filtered_x = []
    filtered_y = []

    for x in to_plot:
        # print to_plot[x]
        values = []
        for c in to_plot[x]:
            # values.extend(c)
            if 1 < len(c) < 10:
                plt.plot([x for _ in c],c,".")

                filtered_x.extend([x for _ in c])
                filtered_y.extend([[i,] for i in c])
    plt.ylim((max(y_pts),0))
    plt.show()

    filtered_x = np.asarray(filtered_x)
    filtered_y = np.asarray(filtered_y)


    db = DBSCAN(eps=1, min_samples=15).fit(filtered_y)

    labels = db.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    print len(unique_labels)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        index_filter = np.where(labels == k)

        x_val = filtered_x[index_filter]
        # print len(x_val)

        # if len(x_val) < 20:
        #     continue

        percentage = len(x_val)/float(max(x_val)-min(x_val))

        # if percentage <0.1:
        #     continue
        #
        # if len(x_val) < 3:
        #     continue

        y_val = [f[0] for f in filtered_y[index_filter]]

        plt.plot(x_val, y_val, 'o', markerfacecolor=col)
    plt.ylim((max(y_pts),0))
    plt.show()

    # for x in range(min(x_pts),max(x_pts)):
    #     print x
    #     id_ = np.where(x_pts==x)
    #     # restricted_y = [[p,] for p in y_pts[id_]]
    #     restricted_y = y_pts[id_]
    #
    #     clusters = [[restricted_y[0],],]
    #     for y in restricted_y[1:]:
    #         if y-clusters[-1][-1] <= 2:
    #             clusters[-1].append(y)
    #         else:
    #             clusters.append([y])
    #
    #     for c in clusters:
    #         plt.plot([x for _ in c],c,"o")
    #     continue
    #
    #     db = DBSCAN(eps=4, min_samples=1).fit(restricted_y)
    #     labels = db.labels_
    #
    #     unique_labels = set(labels)
    #     colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             continue
    #
    #         class_member_mask = (labels == k)
    #         # print class_member_mask
    #
    #
    #         y_cluster = temp_y[class_member_mask]
    #
    #         if len(y_cluster) < 20:
    #             plt.plot([x for _ in range(len(y_cluster))], y_cluster, 'o', markerfacecolor=col)
    #             # xPts.extend([x for _ in range(len(y))])
    #             # yPts.extend(y)
    # plt.show()
    #
    # if display:
    #     plt.plot(x,y,".")
    #     plt.ylim((y_lim,0))
    #     plt.show()
    #
    # return
    #
    # n, bins, patches = plt.hist(y,range(min(y),max(y)+1))
    # med = np.median(n)
    #
    #
    # peaks = [i for i,count in enumerate(n) if count > 2*med]
    # buckets = [[peaks[0]]]
    #
    #
    #
    #
    # for j,p in list(enumerate(peaks))[1:]:
    #     if (p-1) != (peaks[j-1]):
    #         buckets.append([p])
    #     else:
    #         buckets[-1].append(p)
    #
    # bucket_items = list(enumerate(buckets))
    # bucket_items.sort(key = lambda x:len(x[1]),reverse=True)
    #
    # a = bucket_items[0][0]
    # b = bucket_items[1][0]
    #
    #
    # vert = []
    # for x,p in bucket_items:
    #     if (a <= x <= b) or (b <= x <= a):
    #         vert.extend(p)
    #
    # print vert
    # print min(vert)
    # print max(vert)
    #
    # if display:
    #     plt.plot((min(y),max(y)+1),(2*med,2*med))
    #     plt.show()
    #
    #     n, bins, patches = plt.hist(x,range(min(x),max(x)+1))
    #     med = np.median(n)
    #
    #     plt.plot((min(x),max(x)+1),(2*med,2*med))
    #     plt.plot((min(x),max(x)+1),(1*med,1*med),color="red")
    #     plt.xlim((0,max(x)+1))
    #     plt.show()

if __name__ == "__main__":
    # project = aggregation_api.AggregationAPI(153,"development")
    # f_name = project.__image_setup__(1125393)
    f_name = "/home/ggdhines/Databases/images/e1d11279-e515-42f4-a4d9-8e8a40a28425.jpeg"
    analyze(f_name,display=True)