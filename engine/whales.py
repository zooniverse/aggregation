__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.cluster import DBSCAN
import numpy as np
import math
import cv2

# subject_id = 918463
subject_id = 917160
project = AggregationAPI(11,"development")
fname = project.__image_setup__(subject_id)

#478758
#917091

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(list(set(points)))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower

# image_file = cbook.get_sample_data("/home/ggdhines/Databases/images/2fe9c6d0-4b1b-49a4-a96e-15e1cace73b8.jpeg")
image_file = cbook.get_sample_data(fname)
image = plt.imread(image_file)


pts = []

edges = cv2.Canny(image,200,240)

plt.imshow(edges,cmap = 'gray')
plt.show()


sizes = np.shape(image)
height = float(sizes[0])
width = float(sizes[1])
plt.figure(figsize=(height*2/72, width*2/72), dpi=72)

# print len(edges)
for y in range(len(edges)):
    for x in range(len(edges[0])):
        if edges[y][x] > 0:
            # print x,y
            pts.append((x,y))

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
X = np.asarray(pts)
db = DBSCAN(eps=3, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
plt.imshow(image)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))


for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    if len(xy) < 150:
        continue
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=col,markeredgecolor='k', markersize=10)

    p_dict = {}
    for p in list(xy):
        x,y = p.tolist()
        if x in p_dict:
            p_dict[x] = min(p_dict[x],y)
        else:
            p_dict[x] = y

    p_items = p_dict.items()
    notch_x,notch_y = max(p_items, key = lambda x:x[1])


    p_temp = [tuple(p) for p in xy.tolist()]

    upper = convex_hull(p_temp)
    # x,y = zip(*upper)

    if upper != []:
        left = [(x,y) for (x,y) in upper if x < notch_x]
        right = [(x,y) for (x,y) in upper if x > notch_x]

        if left != [] and right != []:
            tip_x,tip_y = min(left, key = lambda x:x[1])
            plt.plot(tip_x,tip_y,'o',color="yellow")
            tip_x,tip_y = min(right, key = lambda x:x[1])
            plt.plot(tip_x,tip_y,'o',color="yellow")

            plt.plot(notch_x,notch_y,'o',color="green")

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelleft='off')

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
plt.xlim((0,width))
plt.ylim((height,0))
plt.savefig("/home/ggdhines/"+str(subject_id)+".jpeg",bbox_inches='tight', pad_inches=0,dpi=72)
# plt.subplot(121),plt.imshow(image,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


