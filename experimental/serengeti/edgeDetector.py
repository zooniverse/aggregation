#!/usr/bin/env python
__author__ = 'greg'

from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE,canny
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.util import img_as_ubyte
from skimage.morphology import disk

img1 = rgb2gray(data.load("/home/greg/Databases/serengeti/blob/50c210188a607540b9000012_0.jpg"))
edges1 = canny(img1)
fig,ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(edges1, cmap=plt.cm.gray)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig("/home/greg/Databases/serengeti/blob/edges.jpg",bbox_inches='tight', pad_inches=0)

fig, ax = plt.subplots(nrows=1, ncols=1)
detector = CENSURE()

img2 = rgb2gray(data.load("/home/greg/Databases/serengeti/blob/edges.jpg"))

# detector.detect(img2)
# ax.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],2 ** detector.scales, facecolors='none', edgecolors='r')
# plt.show()

# image = img_as_ubyte(rgb2gray(data.load("/home/greg/Databases/serengeti/blob/50c210188a607540b9000012_0.jpg")))
#
# fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
#
# img0 = ax0.imshow(image, cmap=plt.cm.gray)
# ax0.set_title('Image')
# ax0.axis('off')
# fig.colorbar(img0, ax=ax0)
#
# img1 = ax1.imshow(entropy(image, disk(5)), cmap=plt.cm.jet)
# ax1.set_title('Entropy')
# ax1.axis('off')
# fig.colorbar(img1, ax=ax1)
#
# plt.show()

