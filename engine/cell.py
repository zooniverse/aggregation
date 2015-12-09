__author__ = 'ggdhines'
import matplotlib
matplotlib.use('WXAgg')
# from aggregation_api import AggregationAPI
import json
# # project = AggregationAPI(1457,"development")
# # project.__image_setup__(1136408)
#
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.cluster import DBSCAN
import numpy as np
#
# image_file = cbook.get_sample_data("/home/ggdhines/Databases/images/2de6e450-7ec3-4f8d-bc29-361314dfacf1.jpeg")
# image = plt.imread(image_file)
#
# pts_x = []
# pts_y = []
#
# image = rgb2gray(image)
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         if image[i][j] < 30:
#             # plt.plot(i,j,color="blue")
#             pts_x.append(i)
#             pts_y.append(j)
# print "here"
# plt.plot(pts_x,pts_y,".",color="blue")
# plt.show()
#
# X = np.asarray(zip(pts_x,pts_y))
# db = DBSCAN(eps=10, min_samples=3).fit(X)
# labels = db.labels_
#
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# print len(unique_labels)
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=3)
#
# plt.show()

directory = "/home/ggdhines/Downloads/tmp/imgs/"
fname = "1080284.jpg"

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
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        continue

    class_member_mask = (labels == k)
    if sum([1 for i in labels if i == k]) > 100:
        continue
    if sum([1 for i in labels if i == k]) < 5:
        continue

    xy = X[class_member_mask]
    x = np.median(xy[:, 0])
    y = np.median(xy[:, 1])
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor="green", markersize=2)
    circle=plt.Circle((x,y),5,color='blue',fill=False)
    ax.add_artist(circle)


print gold_standard[fname]
for x,y,r in gold_standard[fname]:
    # plt.plot(x,y,"o",color="r")
    circle=plt.Circle((x,y),5,color='r',fill=False)
    ax.add_artist(circle)



plt.show()