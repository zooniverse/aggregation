#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
import sqlite3 as lite
import cv2
import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt
import glob
import random
__author__ = 'ggdhines'

warnings.filterwarnings('error')

base = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"
region_bound = (559,3282,1276,2097)

con = lite.connect('/home/ggdhines/to_upload3/active.db')

cur = con.cursor()

bitmaps = []
labels = []

all_heights = []
all_widths = []

# <<<<<<< 8ce8e3a4c7d9552551e469bcbd4315f395525d63
for char in "A1234567890":#BCDEF123456":#CDEFGHIJKLMNOPQRSTUVWXYZ-1234567890.abcdefghijkmnopqrstuvwxyz":
    print(char)
    cur.execute("select * from characters where characters = \"" + char + "\" and confidence > 80 order by confidence desc")

    t = 0

    for r in cur.fetchall()[:150]:
        t += 1
        subject_id, region, column, row, characters, confidence, lb_x, ub_x, lb_y, ub_y = r

        all_heights.append(ub_y-lb_y+1)
        all_widths.append(ub_x-lb_x+1)

        masked_image = cv2.imread("/home/ggdhines/to_upload3/" + subject_id + ".jpg", 0)
        char_image = masked_image[lb_y:ub_y + 1, lb_x:ub_x + 1]

        resized_char = cv2.resize(char_image,(28,28))
        bitmaps.append(resized_char.flatten())
        labels.append(1)

    print(t)

print(len(labels))

files = glob.glob("/home/ggdhines/to_upload3/B*.jpg")
for _ in range(1000):
    fname = files[random.randint(0,len(files)-1)]
    table = cv2.imread(fname,0)



    j = random.randint(0,len(all_heights)-1)


    h = all_heights[j]
    w = all_widths[j]

    shape = table.shape
    row = random.randint(0,shape[0]-h)
    column = random.randint(0,shape[1]-w)

    subimage = table[row:row+h,column:column+w]

    if np.max(subimage) == 0:
        continue

    resized_image = cv2.resize(subimage,(28,28))
    bitmaps.append(resized_image.flatten())
    labels.append(0)

pca = PCA(n_components=15)
X = np.asarray(bitmaps)
X_r = pca.fit(X).transform(X)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
# raw_input("enter something")
n_neighbors = 30
clf = neighbors.KNeighborsClassifier(n_neighbors)

clf.fit(X_r,np.asarray(labels))

desired_h = int(np.percentile(all_heights,90))
desired_w = int(np.percentile(all_widths,90))

def sliding_window(image, stepSize):
    # slide a window across the image
    for y_i,y_value in enumerate(xrange(0, image.shape[0], stepSize)):
        for x_i,x_value in enumerate(xrange(0, image.shape[1], stepSize)):
            # yield the current window
            # print(image,y_value,y_value+h,x_value,x_value + w)
            yield x_value,y_value,image[y_value:y_value + desired_h, x_value:x_value + desired_w]

img = cv2.imread("/home/ggdhines/masked3.jpg",0)
plt.imshow(img,cmap="gray")
total = 0
for x_value,y_value,w in sliding_window(img,2):
    print(y_value)
    resized_char = cv2.resize(w,(28,28))
    x = resized_char.flatten()
    x_r = pca.transform(x)

    p = clf.predict_proba(x_r)[0][1]
    # print(p)
    if p > 0.9:
        plt.plot([x_value,x_value,x_value+desired_w,x_value+desired_w,x_value],[y_value,y_value+desired_h,y_value+desired_h,y_value,y_value],color="blue")
        # print(p)
        # plt.imshow(w)
plt.show()