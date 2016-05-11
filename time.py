from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = np.ones((5,5),np.uint8)

time_series = []

axis = 2

for fname in glob.glob("/home/ggdhines/Databases/images/time_series/*.jpg"):
    img = cv2.imread(fname)[:,:,axis]
    equ = cv2.equalizeHist(img)
    f = equ.astype(float)
    time_series.append(f)

upper_bound = np.percentile(time_series,80,axis=0)
lower_bound = np.percentile(time_series,20,axis=0)

mean_image = np.mean(time_series,axis=0)
cv2.imwrite("/home/ggdhines/github/aggregation/docs/source/images/avg_img.jpg",mean_image.astype(np.uint8))

cv2.imwrite("/home/ggdhines/github/aggregation/docs/source/images/upperbound_img.jpg",upper_bound.astype(np.uint8))


# print(avg_image.shape)

# plt.imshow(avg_image,cmap="gray")
# plt.show()

# avg_image = avg_image.astype(float)

for ii,fname in enumerate(sorted(glob.glob("/home/ggdhines/Databases/images/time_series/*.jpg"))):
    # fname = "/home/ggdhines/Databases/images/time_series/50c212438a607540b901d4ba_0.jpg"
    img = cv2.imread(fname)[:,:,axis]
    # plt.imshow(img)
    # plt.show()

    equ = cv2.equalizeHist(img)
    # plt.imshow(equ)
    # plt.show()

    # t = np.logical_or(f>upper_bound , f < lower_bound)

    template = np.zeros(img.shape,np.uint8)
    t2 = np.where(np.logical_or(equ>upper_bound , equ < lower_bound))
    template[t2] = 255

    # plt.imshow(template)
    # plt.title(fname[-30:])
    # plt.show()

    img = cv2.imread(fname)

    cv2.imwrite("/home/ggdhines/github/aggregation/docs/source/images/"+str(ii)+"_original.jpg",img)

    plt.imshow(img)
    plt.show()

    opening = cv2.morphologyEx(template, cv2.MORPH_OPEN, kernel)

    t = np.where(opening>0)
    for (x,y) in zip(t[0],t[1]):
        img[x,y] = (255,255,0)

    cv2.imwrite("/home/ggdhines/github/aggregation/docs/source/images/"+str(ii)+"_modified.jpg",img)

    plt.imshow(img)
    # plt.imshow(opening)
    # plt.plot(t,"o")
    plt.title(fname[-30:])
    plt.show()

    # plt.imshow(f<lower_bound)
    # plt.title(fname[-30:])
    # plt.show()