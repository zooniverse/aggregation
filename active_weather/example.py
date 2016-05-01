#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from active_weather import __otsu_bin__,__pca__,__create_mask__
import cPickle, gzip
import math

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
# print(n_samples)
# print(train_set[0][0])
# assert False
data = train_set[0].reshape((n_samples, -1))

classifier = svm.SVC(probability=True,gamma=1)

# We learn the digits on the first half of the digits
classifier.fit(data[:5000], train_set[1][:5000])

big_list = []

_,contour, hier = cv2.findContours(masked_image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
for cnt,h in zip(contour,hier[0]):
    perimeter = cv2.arcLength(cnt,True)
    if perimeter < 100:
        continue
    if h[-1] != -1:
        continue

    if len(big_list) == 3:
        break

    cnt = cnt.reshape((cnt.shape[0],2))

    bottom,left = np.min(cnt,axis=0)
    top,right = np.max(cnt,axis=0)

    canvas = np.zeros(table.shape[:2],np.uint8)
    # canvas.fill(255)
    cv2.drawContours(canvas, [cnt], 0, 1, -1)

    # plt.show()

    character = canvas[left:right,bottom:top]
    # plt.imshow(character,cmap="gray")
    # plt.show()

    rescaled = cv2.resize(character,(28,28)).astype(np.float)
    # print(rescaled.shape)
    # plt.imshow(rescaled,cmap="gray")
    # plt.show()

    flattened_character = rescaled.flatten()
    print(flattened_character[:100])
    big_list.append(flattened_character)
    continue

    predicted = classifier.predict([flattened_character])
    probabilities = classifier.predict_proba([flattened_character])
    print(predicted)
    print(probabilities)

    plt.imshow(canvas,cmap="gray")
    plt.show()

print(np.asarray(big_list).shape)
print(classifier.predict_proba(np.asarray(big_list)))


