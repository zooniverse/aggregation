import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris,corner_peaks
from sklearn.cluster import DBSCAN
from skimage.data import load

# filename = "/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG"
# img = cv2.imread(filename)
#
# filename = "/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0185.JPG"
# img2 = cv2.imread(filename)

import Image
mypath = "/home/ggdhines/Databases/old_weather/aligned_images/"
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# print onlyfiles
print mypath+onlyfiles[0]

base_image = Image.open(mypath+onlyfiles[0])
for ii,fname in enumerate(onlyfiles[1:]):
    print 1/(ii+2.)
    image = Image.open(mypath+fname)
    base_image = Image.blend(base_image,image,1/(ii+2.))

base_image.save("/home/ggdhines/new.jpg","JPEG")
img = cv2.imread("/home/ggdhines/new.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh1 = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
cv2.imwrite("/home/ggdhines/t.png",thresh1)


# # background = background.convert("RGBA")
# # overlay = overlay.convert("RGBA")
#
# new_img = Image.blend(background, overlay, 0.5)
# new_img.save("/home/ggdhines/new.JPG","JPEG")
#
# new_img = Image.open("/home/ggdhines/new.JPG")
# new_img = Image.blend(new_img, third, 0.33)
# new_img.save("/home/ggdhines/new.JPG","JPEG")
#
# new_img = Image.open("/home/ggdhines/new.JPG")
# new_img = Image.blend(new_img, fourth, 0.25)
# new_img.save("/home/ggdhines/new.JPG","JPEG")
#
# new_img = Image.open("/home/ggdhines/new.JPG")
# new_img = Image.blend(new_img, fifth, 0.2)
# new_img.save("/home/ggdhines/new.JPG","JPEG")
#
# img = cv2.imread("/home/ggdhines/new.JPG")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh1 = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
# cv2.imwrite("/home/ggdhines/t.png",thresh1)
#
# image = load("/home/ggdhines/t.png")
#
# # print img.shape
# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #
# coords = corner_peaks(corner_harris(image), min_distance=1)
# #
# db = DBSCAN(eps=5, min_samples=2).fit(coords)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# fig, ax1 = plt.subplots(1, 1)
# fig.set_size_inches(52,78)
# ax1.imshow(image)
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print n_clusters_
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#         continue
#
#     class_member_mask = (labels == k)
#
#     xy = coords[class_member_mask]
#     plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor="green", markersize=3)
#
#
#
# # X,Y = zip(*coords)
#
# # plt.plot(Y,X,".",color="red")
# plt.savefig("/home/ggdhines/test.png",bbox_inches='tight', pad_inches=0,dpi=72)