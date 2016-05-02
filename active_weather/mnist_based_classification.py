#!/usr/bin/env python
from __future__ import print_function
from active_weather import __extract_region__,__create_mask__,__otsu_bin__,__pca__,__mask_lines__,__binary_threshold_curry__
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensor import mnist_tensor

table = __extract_region__("/home/ggdhines/eastwind-wag-279-1946_0523-0.JPG",region=(127,1467,275,1151))

pca_image, threshold, inverted = __pca__(table, __binary_threshold_curry__(175))
print(threshold)
mask = __create_mask__(table)

masked_image = __mask_lines__(pca_image,mask)

kernel = np.ones((5,5),np.uint8)
# masked_image = 255 - masked_image
# closing = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel)
# closing = 255 - closing


im2, contours, hierarchy = cv2.findContours(masked_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if (w/h > 5) or (h/w > 5) or (perimeter < 20):
        cv2.drawContours(masked_image, [cnt], 0, 255, -1)
    else:

        print(perimeter)


plt.imshow(masked_image, cmap="gray")
plt.show()
#     perimeter = cv2.arcLength(cnt, True)
#
#     if (min(w,h) > 5) and (perimeter > 50) and (perimeter < 1000):
#         copy_table = table.copy()
#
#
#         print(perimeter)
#
#         bitmap = np.zeros((h,w),np.uint8)
#         big_bitmap = np.zeros(pca_image.shape,np.uint8)
#         cv2.drawContours(copy_table, [cnt], 0, (255,255,0), -1)
#
#         shape = cnt.shape
#         cnt = cnt.reshape((shape[0],shape[2]))
#         cnt -= (x,y)
#
#         assert np.min(cnt) == 0
#
#         cv2.drawContours(bitmap, [cnt], 0, 1, -1)
#
#         plt.imshow(copy_table, cmap="gray")
#         plt.show()
#
#         plt.imshow(bitmap,cmap="gray")
#         plt.show()

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y_i,y_value in enumerate(xrange(0, image.shape[0], stepSize)):
        for x_i,x_value in enumerate(xrange(0, image.shape[1], stepSize)):
            # yield the current window
            yield x_i,y_i,image[y_value:y_value + windowSize[1], x_value:x_value + windowSize[0]]



classifier = mnist_tensor()

for x_i,y_i,window in sliding_window(masked_image,5,(35,25)):
    probability = classifier.__eval__(window)
    if probability < 0.9:
        continue
    # windows = cv2.resize(window,(28,28))
    plt.imshow(window)
    plt.title(str(probability))
    plt.show()