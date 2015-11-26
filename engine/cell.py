__author__ = 'ggdhines'
from aggregation_api import AggregationAPI

# project = AggregationAPI(1457,"development")
# project.__image_setup__(1136408)

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.cluster import DBSCAN
import numpy as np

image_file = cbook.get_sample_data("/home/ggdhines/Databases/images/2de6e450-7ec3-4f8d-bc29-361314dfacf1.jpeg")
image = plt.imread(image_file)

pts_x = []
pts_y = []

image = rgb2gray(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j] < 30:
            # plt.plot(i,j,color="blue")
            pts_x.append(i)
            pts_y.append(j)
print "here"
plt.plot(pts_x,pts_y,".",color="blue")
plt.show()

X = np.asarray(zip(pts_x,pts_y))
db = DBSCAN(eps=10, min_samples=3).fit(X)
labels = db.labels_

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
print len(unique_labels)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=3)

plt.show()





