__author__ = 'greg'
mypath = "/home/greg/Databases/tests/"

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import numpy
import PIL
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) and "jpg" in f]

for fname in onlyfiles:
    print fname
    img = cv2.imread(mypath+fname)

    ref = 101,94,65
    v = []
    for x,a in enumerate(img):
        for y,b in enumerate(a):
            d = math.sqrt(sum([(a1-b1)**2 for a1,b1 in zip(ref,b)]))


            if d < 150:
                # plt.plot(y,-x,'o',color="blue")
                v.append((y,-x))
    # plt.show()
    # continue
    # print img

    # # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    #
    # # noise removal
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    #
    # # sure background area
    # sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    #
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)
    # cv2.imwrite("/home/greg/image_processed.png",thresh)
    #
    # img = PIL.Image.open("/home/greg/image_processed.png").convert("L")
    # arr = numpy.array(img)
    #
    #
    # v = []
    # for i,r in enumerate(arr):
    #     for j,c in enumerate(r):
    #         if c != 0:
    #             v.append((i,j))
    #
    #             plt.plot(i,j,'o')
    # plt.show()


    # print len(v)
    # import numpy as np
    X = np.asarray(v)

    db = DBSCAN(eps=1, min_samples=2).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()