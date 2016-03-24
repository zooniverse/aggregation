#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
from sklearn.decomposition import PCA
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paper_quad
import kernel_smoothing
from sklearn.cluster import DBSCAN

__author__ = 'ggdhines'

table = cv2.imread("/home/ggdhines/region.jpg")
gray = cv2.cvtColor(table,cv2.COLOR_BGR2GRAY)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,2)
ret,th1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)

cv2.imwrite("/home/ggdhines/testing1.jpg",th2)
cv2.imwrite("/home/ggdhines/testing2.jpg",th2)

pca = PCA(n_components=1)
s = table.shape
flatten_table = np.reshape(table,(s[0]*s[1],3))

X_r = pca.fit_transform(flatten_table)
background = max(X_r)[0]
foreground = 0

pca_table = np.reshape(X_r,s[:2])

# fig, ax = plt.subplots()
# cax = ax.imshow(pca_table,cmap="gray")
# cbar = fig.colorbar(cax)
plt.imshow(pca_table[300:450,2000:2100])
plt.show()

s = pca_table.shape
f = np.reshape(pca_table,(s[0]*s[1]))
plt.hist(f, bins=100)
plt.show()
assert False
# cv2.imwrite("/home/ggdhines/pca.jpg",pca_table)
# assert False
# print(pca_table)
# ink_pixels = np.where(pca_table > 0)
# plt.plot(ink_pixels[1],-ink_pixels[0],".")
# plt.show()

gray = cv2.cvtColor(table,cv2.COLOR_BGR2GRAY)

horizontal_lines = paper_quad.__extract_grids__(gray,True)

mask = np.zeros(gray.shape,np.uint8)
for l in horizontal_lines:
    corrected_l = kernel_smoothing.__correct__(gray,l,True)
    mask = np.max([mask,corrected_l],axis=0)

cv2.imwrite("/home/ggdhines/testing.jpg",mask)
vertical_lines = paper_quad.__extract_grids__(gray,False)
for l in vertical_lines:
    corrected_l = kernel_smoothing.__correct__(gray,l,False)
    mask = np.max([mask,corrected_l],axis=0)

mask = 255 - mask
# plt.imshow(mask,cmap="gray")
# plt.show()

t = np.min([pca_table,mask],axis=0)

# plt.imshow(t)
# plt.show()

ink_pixels = np.where(t > 0)
plt.plot(ink_pixels[1],-ink_pixels[0],".")
plt.show()
# assert False

X = np.asarray(zip(ink_pixels[1],ink_pixels[0]))
print("doing dbscan: " + str(X.shape))
db = DBSCAN(eps=1, min_samples=5).fit(X)

labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

return_image = np.zeros(gray.shape,np.uint8)
return_image.fill(255)

print("going through dbscan results")
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        continue

    class_member_mask = (labels == k)
    # temp = np.zeros(X.shape)

    xy = X[class_member_mask]

    max_value = gray[xy[:, 1], xy[:, 0]].max()
    median = np.median(gray[xy[:, 1], xy[:, 0]])
    mean = np.mean(gray[xy[:, 1], xy[:, 0]])
    # print(max_value,median,mean)

    if True:#median > 120:
        x_max,y_max = np.max(xy,axis=0)
        x_min,y_min = np.min(xy,axis=0)
        if min(x_max-x_min,y_max-y_min) >= 10:
            return_image[xy[:, 1], xy[:, 0]] = gray[xy[:, 1], xy[:, 0]]

cv2.imwrite("/home/ggdhines/hello_world.jpg",return_image)
plt.imshow(return_image,cmap="gray")
plt.show()