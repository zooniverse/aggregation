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

print image_directory+'eastwind-wag-279-1946_0149-0.JPG'
print image_directory+'eastwind-wag-279-1946_0382-0.JPG'

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

image1_pts = []
image2_pts = []

dimensions =  img1.shape


for m in matches[:20]:
    i1,i2 = m.queryIdx,m.trainIdx

    p1,p2 = kp1[i1].pt,kp2[i2].pt

    image1_pts.append((p1[0],p1[1],1))
    image2_pts.append((p2[0],p2[1],1))






# https://www.cis.rit.edu/class/simg782/lectures/lecture_02/lec782_05_02.pdf

sz = img1.shape
P,Q,H = align(image1_pts,image2_pts)
for p,q in zip(P.transpose(),Q.transpose()):

    # print (p[(0,0)],p[(0,1)]),(q[(0,0)],q[(0,1)])
    (p_x,p_y),(q_x,q_y) = (p[(0,0)],p[(0,1)]),(q[(0,0)],q[(0,1)])
    old_distance = math.sqrt((p_x-q_x)**2+(p_y-q_y)**2)
    # print H
    # print
    t = H*p.transpose()

    t_x,t_y = t[(0,0)],t[(1,0)]
    new_distance = math.sqrt((t_x-q_x)**2+(t_y-q_y)**2)
    print old_distance,new_distance
# assert False

    # print math.sqrt((p[(0,0)]-q[(0,0)])**2+(p[(0,1)]-q[(0,1)])**2)
# H = np.squeeze(np.asarray(H)[:2])
# print H
H = np.asarray(H)[:2]
# print H
im2_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
# im2_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)

X,Y,W = im2_aligned.shape
for x in range(X):
    for y in range(Y):
        for w in range(W):
            im2_aligned[(x,y,w)] = (im2_aligned[(x,y,w)] + img1[(x,y,w)])/2

cv2.imwrite("/home/ggdhines/t.png",im2_aligned)

# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 1", img2)
# cv2.imshow("Aligned Image 2", im2_aligned)
# cv2.waitKey(0)
#sel_matches = [m for m in matches if m.distance <= 50]
# Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,sel_matches, None,flags=2)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],i)

# plt.imshow(img3)
# plt.savefig("/home/ggdhines/t.png",dpi=500)
# plt.show()




# warp_mode = cv2.MOTION_EUCLIDEAN
# number_of_iterations = 5000
#
# # Specify the threshold of the increment
# # in the correlation coefficient between two iterations
# termination_eps = 1e-10
#
# # Define termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
# warp_matrix = np.eye(2, 3, dtype=np.float32)
# # Run the ECC algorithm. The results are stored in warp_matrix.
# # (cc, warp_matrix) = cv2.findTransformECC(temp_image1,temp_image2,warp_matrix, warp_mode, criteria)
# (cc, warp_matrix) = cv2.findTransformECC(img1_gray,img2_gray,warp_matrix, warp_mode, criteria)
# sz = img1.shape
# im2_aligned = cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
#
# # Show final results
# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 2", img2)
# cv2.imshow("Aligned Image 2", im2_aligned)
# cv2.waitKey(0)