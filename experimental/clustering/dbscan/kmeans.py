#!/usr/bin/env python
__author__ = 'greghines'

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pts =[]

with open('/Users/greghines/Databases/gps/smallGPS','r') as f:
    for line in f.readlines():
        x,y = line[:-1].split(',')
        pts.append((float(y)/pow(10,7),float(x)/pow(10,7)))

pts = np.array(pts)


print type(pts)

kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
kmeans.fit(pts)
#db = DBSCAN(eps=5, min_samples=20).fit(pts)

h=0.2
x_min, x_max = pts[:, 0].min() + 1, pts[:, 0].max() - 1
y_min, y_max = pts[:, 1].min() + 1, pts[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.show()