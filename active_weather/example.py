#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from active_weather import __otsu_bin__,__pca__,__create_mask__
import cPickle, gzip

img = cv2.imread("/home/ggdhines/eastwind-wag-279-1946_0523-0.JPG")

table = img[276:1158,126:1476]
# plt.imshow(table)
# plt.show()

cv2.imwrite("/home/ggdhines/table.jpg",table)

pca_image,threshold,inverted = __pca__(table,__otsu_bin__)

cv2.imwrite("/home/ggdhines/pca_example.jpg",pca_image)

mask = __create_mask__(table)

cv2.imwrite("/home/ggdhines/mask_example.jpg",mask)

masked_image = np.max([mask,pca_image],axis=0)

plt.imshow(masked_image,cmap="gray")
plt.show()

# from sklearn.datasets import fetch_mldata
from sklearn import datasets, svm, metrics



# Load the dataset
f = gzip.open('/home/ggdhines/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
print(train_set)
n_samples = len(train_set[0])
print(n_samples)
data = train_set[0].reshape((n_samples, -1))

classifier = svm.SVC(probability=True,gamma=0.001)

# We learn the digits on the first half of the digits
# classifier.fit(data[:5000], train_set[1][:5000])

_,contour, hier = cv2.findContours(masked_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
for cnt in contour:
    perimeter = cv2.arcLength(cnt,True)
    if perimeter < 100:
        continue
    print(perimeter)
    cnt = cnt.reshape((cnt.shape[0],2))

    bottom,left = np.min(cnt,axis=0)

    # cnt = cnt - (bottom,left)

    top,right = np.max(cnt,axis=0)
    canvas = np.zeros(table.shape,np.uint8)
    canvas.fill(255)
    cv2.drawContours(canvas, [cnt], 0, 0, -1)
    plt.imshow(canvas,cmap="gray")
    plt.show()



