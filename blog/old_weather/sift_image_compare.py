import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import Image
from numpy.linalg import inv
import matplotlib.cbook as cbook


def align(p_list1,p_list2):
    assert len(p_list1) == len(p_list2)
    P = np.mat(np.array(p_list1)).transpose()
    Q = np.mat(np.array(p_list2)).transpose()
    s = P*(P.transpose())
    s_inv = inv(s)
    P_dagger = P.transpose()*s_inv

    # print P_dagger
    H = Q*P_dagger

    return P,Q,H

image_directory = "/home/ggdhines/Databases/old_weather/images/"

fname1 = image_directory+'eastwind-wag-279-1946_0149-0.JPG'
fname2 = image_directory+'eastwind-wag-279-1946_0382-0.JPG'
img1 = cv2.imread(fname1)          # queryImage
img2 = cv2.imread(fname2) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []

image1_pts = []
image2_pts = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        i1,i2 = m.queryIdx,m.trainIdx

        p1,p2 = kp1[i1].pt,kp2[i2].pt

        dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

        if dist < 100:
            image1_pts.append((p1[0],p1[1],1))
            image2_pts.append((p2[0],p2[1],1))

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
image_file = cbook.get_sample_data(fname1)
image = plt.imread(image_file)
# fig, ax = plt.subplots()
im = axes.imshow(image)
for x,y,_ in image1_pts:
    plt.plot(x,y,"o",color="green",markersize=1)
plt.savefig("/home/ggdhines/a.png",dpi=1000)
plt.close()

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
image_file = cbook.get_sample_data(fname2)
image = plt.imread(image_file)
# fig, ax = plt.subplots()
im = axes.imshow(image)
for x,y,_ in image2_pts:
    plt.plot(x,y,"o",color="green",markersize=1)
plt.savefig("/home/ggdhines/a_.png",dpi=1000)
plt.close()
sz = img1.shape
P,Q,H = align(image1_pts,image2_pts)

image1_pts_take2 = []
image2_pts_take2 = []

for p,q in zip(P.transpose(),Q.transpose()):

    # print (p[(0,0)],p[(0,1)]),(q[(0,0)],q[(0,1)])
    (p_x,p_y),(q_x,q_y) = (p[(0,0)],p[(0,1)]),(q[(0,0)],q[(0,1)])
    old_distance = math.sqrt((p_x-q_x)**2+(p_y-q_y)**2)
    # print H
    # print
    t = H*p.transpose()

    t_x,t_y = t[(0,0)],t[(1,0)]
    new_distance = math.sqrt((t_x-q_x)**2+(t_y-q_y)**2)
    if new_distance < old_distance:
        image1_pts_take2.append((p_x,p_y,1))
        image2_pts_take2.append((q_x,q_y,1))
    else:
        print "****"

P,Q,H = align(image1_pts_take2,image2_pts_take2)

H = np.asarray(H)[:2]
im1_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]),cv2.WARP_INVERSE_MAP)#, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
X,Y,W = im1_aligned.shape
for x in range(X):
    for y in range(Y):
        for w in range(W):
            im1_aligned[(x,y,w)] = (im1_aligned[(x,y,w)] + img2[(x,y,w)])/2

cv2.imwrite("/home/ggdhines/t.png",im1_aligned)