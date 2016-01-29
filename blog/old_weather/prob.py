from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

img = cv2.imread("/home/ggdhines/Databases/old_weather/cells/Bear-AG-29-1939-0185_1_7.png")

print img.shape

colours = {}

# under the assumption that most of the cell is not ink - find the most common pixel colour
# any pixel that is "far" enough away is assumed to be ink
for c in range(img.shape[1]):
    for r in range(img.shape[0]):
        pixel_colour = tuple(img[r,c])

        if pixel_colour not in colours:
            colours[pixel_colour] = 1
        else:
            colours[pixel_colour] += 1

most_common_colour,_ = sorted(colours.items(),key = lambda x:x[1],reverse=True)[0]

image = np.zeros(img.shape[:2])
for c in range(img.shape[1]):
    for r in range(img.shape[0]):
        pixel_colour = tuple(img[r,c])
        dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(pixel_colour,most_common_colour)]))

        if dist > 40:
            image[(r,c)] = 150

lines = probabilistic_hough_line(image, threshold=0, line_length=70,line_gap=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_axis_off()
ax1.set_adjustable('box-forced')

# ax2.imshow(image, cmap=plt.cm.gray)
# ax2.set_title('Canny edges')
# ax2.set_axis_off()
# ax2.set_adjustable('box-forced')

ax2.imshow(image)

for line in lines:
    p0, p1 = line
    ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))

ax2.set_title('Probabilistic Hough')
ax2.set_axis_off()
ax2.set_adjustable('box-forced')
plt.show()