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

# img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# # cv2.imshow("Image 1", img1_gray)
# print type(img1_gray)
# print img1_gray.shape

temp_image1 = np.zeros(img1.shape[:2])
temp_image2 = np.zeros(img1.shape[:2])

print type(temp_image1)
print temp_image1.shape

# scipy.misc.imsave('/tmp/outfile.jpg', temp_image)

# plt.imshow(temp_image)
# plt.savefig("/tmp/t.png")
# plt.close()

dimensions =  img1.shape


# Initiate SIFT detector
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
print type(matches)
assert False
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]
print "here here"
print len(matches)
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):

    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
    else:
        print m.queryIdx,

assert False
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.savefig("/home/ggdhines/a.png",dpi=1000)
assert False

for m in matches:
    i1,i2 = m.queryIdx,m.trainIdx

    p1,p2 = kp1[i1].pt,kp2[i2].pt

    dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    if dist < 100:
        print (p1[0],p1[1]),(p2[0],p2[1])
        image1_pts.append((p1[0],p1[1],1))
        image2_pts.append((p2[0],p2[1],1))
        # temp_image1[(p1[1],p1[0])] = 50
        # temp_image2[(p2[1],p2[0])] = 50
        # sel_matches.append(m)

# fig = plt.figure()
# axes = fig.add_subplot(1, 1, 1)
# image_file = cbook.get_sample_data(fname1)
# image = plt.imread(image_file)
# # fig, ax = plt.subplots()
# im = axes.imshow(image)
# for x,y,_ in image1_pts:
#     plt.plot(x,y,"o",color="green",markersize=1)
# plt.axis('off')
# plt.savefig("/home/ggdhines/i1.jpg",dpi=1000,bbox_inches='tight',pad_inches=0.3)
#
#
# fig = plt.figure()
# axes = fig.add_subplot(1, 1, 1)
# image_file = cbook.get_sample_data(fname2)
# image = plt.imread(image_file)
# # fig, ax = plt.subplots()
# im = axes.imshow(image)
# for x,y,_ in image2_pts:
#     plt.plot(x,y,"o",color="green",markersize=1)
# plt.axis('off')
# plt.savefig("/home/ggdhines/i2.jpg",dpi=1000,bbox_inches='tight',pad_inches=0.3)
#
# assert False

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
assert False

    # print math.sqrt((p[(0,0)]-q[(0,0)])**2+(p[(0,1)]-q[(0,1)])**2)
# H = np.squeeze(np.asarray(H)[:2])
print H
H = np.asarray(H)[:2]
print H
im2_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
# im2_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)

X,Y,W = im2_aligned.shape
for x in range(X):
    for y in range(Y):
        for w in range(W):
            im2_aligned[(x,y,w)] = (im2_aligned[(x,y,w)] + img2[(x,y,w)])/2

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