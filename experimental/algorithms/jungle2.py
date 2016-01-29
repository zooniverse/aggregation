__author__ = 'ggdhines'
import aggregation_api
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# project = aggregation_api.AggregationAPI(153,"development")
# f_name = project.__image_setup__(1125393)
f_name = "/home/ggdhines/Databases/images/b828fa18-89b8-4941-903f-0ef27eab7be6.jpeg"
image = cv2.imread(f_name)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

# cv2.imwrite("/home/ggdhines/jungle.jpeg",gray_image)

# print gray_image.ravel()
ix = np.in1d(gray_image.ravel(),range(170)).reshape(gray_image.shape)
ix = np.transpose(ix)

# vert = [568, 569, 570, 571, 572, 573, 574, 575, 576, 139, 140, 141, 142, 143, 144, 145, 146, 354, 355, 356, 357, 534, 535, 536, 537, 177, 178, 179, 319, 320, 321, 390, 391, 392, 426, 427, 428, 461, 462, 463, 497, 498, 499, 212, 213, 248, 249, 283, 284]

# for x in range(ix.shape[0]):
xPts = []
yPts = []
for x in range(139,576):

    pixels = np.where(ix[x])[0]
    pixels = np.asarray([[p,] for p in pixels])
    # plt.plot([x for _ in range(len(pixels))],pixels,".")
    db = DBSCAN(eps=4, min_samples=1).fit(pixels)

    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        class_member_mask = (labels == k)

        y = [p[0] for p in pixels[class_member_mask]]

        if len(y) < 20:
            # plt.plot([x for _ in range(len(y))], y, 'o', markerfacecolor=col)
            xPts.extend([x for _ in range(len(y))])
            yPts.extend(y)


# plt.show()
yPts_ = np.asarray([[p,] for p in yPts])
yPts = np.asarray(yPts)
xPts = np.asarray(xPts)
db = DBSCAN(eps=0.5, min_samples=1).fit(yPts_)
labels = db.labels_
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        continue

    class_member_mask = (labels == k)
    ix = np.in1d(yPts,[k])
    indices = np.where(ix)

    new_x = xPts[indices]
    new_y = yPts[indices]

    plt.plot(new_x,new_y,"o",color=col)

plt.show()

