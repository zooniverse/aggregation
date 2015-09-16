#!/usr/bin/env python
__author__ = 'greg'

from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img1 = rgb2gray(data.load("/home/greg/Databases/serengeti/blob/50c210188a607540b9000004_0.jpg"))
tform = tf.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
                           translation=(150, -200))
img2 = tf.warp(img1, tform)

detector = ORB( non_max_threshold=0.05)

fig, ax = plt.subplots(nrows=1, ncols=1)

plt.gray()

detector.detect(img1)
print detector.scales

ax.imshow(img1)
ax.axis('off')
ax.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')



plt.show()
