import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import Image
from numpy.linalg import inv
import matplotlib.cbook as cbook


def euclidean_least_squares(p_list1,p_list2):
    assert len(p_list1) == len(p_list2)
    P = np.mat(np.array(p_list1)).transpose()
    Q = np.mat(np.array(p_list2)).transpose()
    s = P*(P.transpose())
    s_inv = inv(s)
    P_dagger = P.transpose()*s_inv

    # print P_dagger
    H = Q*P_dagger

    return P,Q,H

def prune_bad_matches__(P,Q,H,thres):
    pruned_pts1 = []
    pruned_pts2 = []

    bad = 0

    for p,q in zip(P.transpose(),Q.transpose()):
        # print (p[(0,0)],p[(0,1)]),(q[(0,0)],q[(0,1)])
        (p_x,p_y),(q_x,q_y) = (p[(0,0)],p[(0,1)]),(q[(0,0)],q[(0,1)])
        old_distance = math.sqrt((p_x-q_x)**2+(p_y-q_y)**2)
        # print H
        # print
        t = H*p.transpose()

        t_x,t_y = t[(0,0)],t[(1,0)]
        new_distance = math.sqrt((t_x-q_x)**2+(t_y-q_y)**2)
        if new_distance < (old_distance*thres): #0.4
            pruned_pts1.append((p_x,p_y,1))
            pruned_pts2.append((q_x,q_y,1))
        else:
            bad += 1
    print " " + str(len(pruned_pts1))
    return pruned_pts1,pruned_pts2,bad


def align(img_fname1,img_fname2):
    img1 = cv2.imread(img_fname1)          # queryImage
    img2 = cv2.imread(img_fname2)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test

    image1_pts = []
    image2_pts = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            i1,i2 = m.queryIdx,m.trainIdx

            p1,p2 = kp1[i1].pt,kp2[i2].pt

            dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

            if dist < 500:
                image1_pts.append((p1[0],p1[1],1))
                image2_pts.append((p2[0],p2[1],1))
    # first pass
    P,Q,H = euclidean_least_squares(image1_pts,image2_pts)

    thresholds = [0.7,0.4,0.3]
    t_i = 0

    for i in range(10):
        # second pass - remove any bad matches
        image1_pts,image2_pts,bad_count = prune_bad_matches__(P,Q,H,thresholds[t_i])
        P,Q,H = euclidean_least_squares(image1_pts,image2_pts)
        print bad_count
        if bad_count == 0:
            t_i += 1

            if t_i == len(thresholds):
                break


    # prune_bad_matches__(P,Q,H)

    H = np.asarray(H)[:2]
    sz = img1.shape

    im1_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]),cv2.WARP_INVERSE_MAP)
    # cv2.imwrite("/home/ggdhines/t.png",im1_aligned)
    # X,Y,W = im1_aligned.shape
    # X_2,Y_2,_ = img2.shape
    # for x in range(X):
    #     if x >= X_2:
    #         continue
    #     for y in range(Y):
    #         if y >= Y_2:
    #             continue
    #         for w in range(W):
    #             im1_aligned[(x,y,w)] = (int(im1_aligned[(x,y,w)]) + int(img2[(x,y,w)]))/2
    #
    # cv2.imwrite("/home/ggdhines/t1.png",im1_aligned)

    return im1_aligned