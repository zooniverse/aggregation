#!/usr/bin/env python
from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.cluster import DBSCAN
import numpy as np

def get_window_size():
    non_white_points = np.where(img[:, :500] != 255)
    non_white_points = np.asarray(zip(non_white_points[0], non_white_points[1]))
    print(non_white_points.shape)
    db = DBSCAN(eps=1, min_samples=5).fit(non_white_points)
    labels = db.labels_

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    print("here")
    heights = []
    widths = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue
        # print(k)

        class_member_mask = (labels == k)
        xy = non_white_points[class_member_mask]
        #
        min_y, min_x = np.min(xy, axis=0)
        max_y, max_x = np.max(xy, axis=0)
        if min(max_x - min_x, max_y - min_y) <= 1:
            continue

        heights.append(max_y - min_y)
        widths.append(max_x - min_x)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y_i,y_value in enumerate(xrange(0, image.shape[0], stepSize)):
        for x_i,x_value in enumerate(xrange(0, image.shape[1], stepSize)):
            # yield the current window
            yield x_i,y_i,image[y_value:y_value + windowSize[1], x_value:x_value + windowSize[0]]

height = 35#int(np.median(heights))
width = 29#int(np.median(widths))

img = cv2.imread("/home/ggdhines/test2.jpg",0)



flatten_windows = []

for x_i,y_i,window in sliding_window(img,10,(width,height)):
    try:
        flatten_windows.append(np.reshape(window,height*width))
    except ValueError:
        pass

print(len(flatten_windows))


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X = np.asarray(flatten_windows)
X_r = pca.fit(X).transform(X)

print(str(sum(pca.explained_variance_ratio_)))
