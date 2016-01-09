import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import DBSCAN

files = ["/home/greg/widlebeest/e614851a-17aa-4499-b410-f2e9985d5e1c.jpeg","/home/greg/widlebeest/79e79bc2-216e-4c85-b08d-49ffd8cdfc49.jpeg","/home/greg/widlebeest/1434512e-e3b1-4da7-837d-bf9998d8412d.jpeg"]
files2 = ["/home/greg/widlebeest/ee7afd20-ef59-4b37-b159-fdb59a79fa27.jpeg","/home/greg/widlebeest/b75e515e-ecf7-4d55-bf7b-fe6bda2741ed.jpeg","/home/greg/widlebeest/758d075b-ee3e-4608-bf82-a672e109e5c1.jpeg"]

files3 = ["/home/greg/widlebeest/8dc3e121-b169-4951-92df-31a28ae623a9.jpeg","/home/greg/widlebeest/79be875a-390e-41cc-948d-cfd0ff5170d4.jpeg","/home/greg/widlebeest/c780d0ee-853d-451a-b9cb-5b0da9655508.jpeg"]

f0 = cv2.imread(files3[0])
f1 = cv2.imread(files3[1])
f2 = cv2.imread(files3[2])


X= []
Y= []

def dist(c1,c2):
    c1 = [int(c) for c in c1]
    c2 = [int(c) for c in c2]
    return math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)

for i in range(len(f0)):
    print i
    for j in range(len(f0[0])):
        if max(dist(f1[i][j],f0[i][j]),0) > 120:#dist(f1[i][j],f2[i][j])) > 125:
            X.append(j)
            Y.append(-i)

XY= np.array(zip(X,Y))
db = DBSCAN(eps=15, min_samples=10).fit(XY)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = XY[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = XY[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
