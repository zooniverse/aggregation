import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris,corner_peaks
from sklearn.cluster import DBSCAN

filename = "/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG"
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

coords = corner_peaks(corner_harris(gray), min_distance=1)

db = DBSCAN(eps=5, min_samples=2).fit(coords)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(52,78)
ax1.imshow(img)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print n_clusters_
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        continue

    class_member_mask = (labels == k)

    xy = coords[class_member_mask]
    plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor="green", markersize=3)



# X,Y = zip(*coords)

# plt.plot(Y,X,".",color="red")
plt.savefig("/home/ggdhines/test.png",bbox_inches='tight', pad_inches=0,dpi=72)