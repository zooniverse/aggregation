import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

img1 = cv2.imread('/home/ggdhines/Databases/old_weather/images/Bear-AG-29-1939-0199.JPG',0)          # queryImage
img2 = cv2.imread('/home/ggdhines/Databases/old_weather/images/Bear-AG-29-1939-0193.JPG',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
sel_matches = []
for m in matches:
    i1,i2 = m.queryIdx,m.trainIdx

    p1,p2 = kp1[i1].pt,kp2[i2].pt

    dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    if dist < 30:
        sel_matches.append(m)

#sel_matches = [m for m in matches if m.distance <= 50]
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,sel_matches, None,flags=2)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],i)

print matches[0].queryIdx

plt.imshow(img3)
plt.savefig("/home/ggdhines/t.png",dpi=500)
plt.show()

#
#
# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
#
# # Apply ratio test
# good = []
#
# ratios = []
#
# for m,n in matches:
#     print m,n
#     assert False
#     ratios.append(n.distance/m.distance)
#
# threshold = np.percentile(ratios,99)
#
# for m,n in matches:
#     if n.distance/m.distance >= threshold:
#     # if m.distance < 0.1*n.distance:
#         good.append([m])
#
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#
# plt.imshow(img3),plt.show()