import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890."
characters = [c for c in characters]
print characters

image = cv2.imread("weather.basic.exp0.tif")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_,bw_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

pixel_points = np.where(bw_image<255)

# plt.plot(pixel_points[1],pixel_points[0],".")
# plt.show()

X = np.asarray(zip(pixel_points[1],pixel_points[0]))

db = DBSCAN(eps=3, min_samples=4).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

clusters = []

print n_clusters_
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):

    class_member_mask = (labels == k)
    xy = X[class_member_mask]

    min_x,min_y = np.min(xy,axis=0)
    max_x,max_y = np.max(xy,axis=0)

    height = max_y-min_y+1
    width = max_x-min_x +1

    clusters.append(xy)

clusters.sort(key = lambda c:np.mean(c[:,0]))

plt.plot(clusters[0][:,0],clusters[0][:,1],".")
plt.show()
