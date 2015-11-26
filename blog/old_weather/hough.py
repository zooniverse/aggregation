from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data
from skimage.data import load
from skimage.color import rgb2gray
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# image = data.camera()
image = load("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG")
# image = rgb2gray(image)

# img = cv2.imread("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG",0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image,50,150,apertureSize = 3)
# cv2.imwrite('/home/ggdhines/1.jpg',edges)

lines = probabilistic_hough_line(edges, threshold=10, line_length=50,line_gap=0)
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(52,78)
ax1.imshow(edges * 0)

for line in lines:
    p0, p1 = line
    ax1.plot((p0[0], p1[0]), (p0[1], p1[1]))

ax1.set_title('Probabilistic Hough')
ax1.set_axis_off()
ax1.set_adjustable('box-forced')
plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
# print type(img)
# print type(image)
# print img.shape,image.shape
# # edges = canny(image, 2, 1, 25)
# _,image = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# print image
#
# pts = []
#
# for x in range(image.shape[0]):
#     for y in range(image.shape[1]):
#         if image[x][y] > 0:
#             pts.append((x,y))
#
#
# tree = KDTree(np.asarray(pts))
# # lines = probabilistic_hough_line(image, threshold=10, line_length=150,line_gap=0)