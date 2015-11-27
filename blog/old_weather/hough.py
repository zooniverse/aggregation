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
import math

def hesse_line(line_seg):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """


    (x1,y1),(x2,y2) = line_seg

    # x2 += random.uniform(-0.0001,0.0001)
    # x1 += random.uniform(-0.0001,0.0001)

    dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

    try:
        tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
        theta = math.atan(tan_theta)
    except ZeroDivisionError:
        theta = math.pi/2.

    return dist,theta


# image = data.camera()
image = load("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG")
# image = rgb2gray(image)

# img = cv2.imread("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG",0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
# cv2.imwrite('/home/ggdhines/1.jpg',edges)

lines = probabilistic_hough_line(edges, threshold=10, line_length=50,line_gap=0)
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(52,78)
# ax1.imshow(image)

hesse_list = []
pruned_lines = []

for line in lines:
    p0, p1 = line
    X = p0[0],p1[0]
    Y = p0[1],p1[1]

    if (min(X) >= 559) and (max(X) <= 3245) and (min(Y) >= 1292) and (max(Y) <= 2014):
        d,t = hesse_line(line)
        if math.fabs(t) <= 0.1:
            pruned_lines.append(line)
            hesse_list.append(hesse_line(line))
        # ax1.plot(X, Y,color="red")
    # else:
    #     print min(X)
    # print hesse_line(line)

from sklearn.cluster import DBSCAN
d_list,t_list = zip(*hesse_list)
min_dist = min(d_list)
max_dist = max(d_list)
# min_theta = min(t_list)
# max_theta = max(t_list)

normalized_d = [[(d-min_dist)/float(max_dist-min_dist),] for d in d_list]
# normalized_t = [(t-min_theta)/float(max_theta-min_theta) for t in t_list]
# print normalized_d
# print normalized_t

X = np.asarray(normalized_d)
print X
db = DBSCAN(eps=0.02, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        continue

    class_indices = [i for (i,l) in enumerate(labels) if l == k]
    # print class_indices

    for i in class_indices:
        line = pruned_lines[i]
        p0, p1 = line
        X = p0[0],p1[0]
        Y = p0[1],p1[1]
        # print X,Y
        ax1.plot(X, Y,color=col)

# ax1.set_title('Probabilistic Hough')
ax1.set_axis_off()
ax1.set_adjustable('box-forced')
plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
plt.close()
#
# h, theta, d = hough_line(edges)
# fig, ax1 = plt.subplots(1, 1)
# # fig.set_size_inches(52,78)
# rows, cols,_ = image.shape
# ax1.imshow(image)
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     print angle
#     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#     y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
#     ax1.plot((0, cols), (y0, y1), '-r')
# plt.show()

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