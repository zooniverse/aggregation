"""
aligns an image to the template in the reference file. The trick is figuring out what the template is (ie. what the
images would like without any data entered on them).
"""
import numpy as np
import cv2
import math
from numpy.linalg import inv
import sys


def euclidean_least_squares(p_list1,p_list2):
    """
    assumes that p_list2 is the result of an Euclidean transformation of the points in p_list1
    Euclidean transformation allows for rotation, translation and magnification
    http://www.cs.cf.ac.uk/Dave/CM2202/LECTURES/CM2202_Geometric_Computing_Trans_Fitting.pdf
    H is the least squares matrix for the transformation
    P,Q are the original points in numpy matrix format
    :param p_list1:
    :param p_list2:
    :return:
    """
    assert len(p_list1) == len(p_list2)
    P = np.mat(np.array(p_list1)).transpose()
    Q = np.mat(np.array(p_list2)).transpose()
    s = P*(P.transpose())
    s_inv = inv(s)
    P_dagger = P.transpose()*s_inv

    H = Q*P_dagger

    return P,Q,H


def prune_bad_matches__(P,Q,H,thres):
    """
    consider the mapping H(P) - where H is an Euclidean mapping determined by least squares fitting
    based on P and Q - if there is no error, i.e. H(P) = Q, then the mapping exactly describes the difference
    between the two images. However, this mapping only applies to points that are part of the template. For a pair of
    points in PxQ, which are not in the template, if they are close in coordinates in the two images this is purely
    by chance - the relationship does not follow the Euclidean mapping. So look for points which H(p)-q is larger than
    thres*(p-q) - i.e. the mapping H does not do a good job of describing the relationship between those two
    points - under the assumption that such points are not part of the template, throw them out
    :param P:
    :param Q:
    :param H:
    :param thres:
    :return:
    """
    pruned_pts1 = []
    pruned_pts2 = []

    bad = 0

    # go through each pair of mapped points
    for p,q in zip(P.transpose(),Q.transpose()):
        # we have points in homogeneous coordinate system - so only take the first two points
        try:
            p_x,p_y = p[(0,0)],p[(0,1)]
            q_x,q_y = q[(0,0)],q[(0,1)]
        except IndexError:
            print P
            print Q
            raise

        # what was the original distance?
        old_distance = math.sqrt((p_x-q_x)**2+(p_y-q_y)**2)

        # what is the distance after the mapping?
        t = H*p.transpose()
        t_x,t_y = t[(0,0)],t[(1,0)]
        new_distance = math.sqrt((t_x-q_x)**2+(t_y-q_y)**2)

        # is the new distance better than the old?
        # the 0.000001 allows for both distances are 0 - useful for when
        # mapping an image to itself
        if new_distance <= ((old_distance*thres)+0.000001):
            pruned_pts1.append((p_x,p_y,1))
            pruned_pts2.append((q_x,q_y,1))
        else:
            bad += 1
    return pruned_pts1,pruned_pts2,bad


def align(img_fname1,img_fname2):
    img1 = cv2.imread(img_fname1)          # queryImage
    img2 = cv2.imread(img_fname2)

    # detector to find matches between two images
    # both SIFT and SURF are possible - for differences see
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html#py-table-of-content-feature2d
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
    # for a walk through of the following code
    index_params = dict(algorithm = 1, trees = 4)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

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
    # iteratively reduce the threshold
    thresholds = [0.7,0.4,0.3,0.15,0.05,0.01]
    t_i = 0

    for i in range(10):
        # second pass - remove any bad matches
        print len(image1_pts),len(image2_pts)
        image1_pts,image2_pts,bad_count = prune_bad_matches__(P,Q,H,thresholds[t_i])
        P,Q,H = euclidean_least_squares(image1_pts,image2_pts)
        # if we have removed all bad points for this given threshold - move on to the next
        if bad_count == 0:
            t_i += 1

            if t_i == len(thresholds):
                break

    # warpAffine - does not use the homogeneous coordinates - so get the first two
    H = np.asarray(H)[:2]
    sz = img1.shape

    # actually align the images
    im1_aligned = cv2.warpAffine(img1, H, (sz[1],sz[0]),cv2.WARP_INVERSE_MAP)

    return len(matches),im1_aligned

if __name__ == "__main__":
    directory = sys.argv[1]
    to_be_aligned = sys.argv[2]
    template = sys.argv[3]
    ship = sys.argv[4]
    year = sys.argv[5]

    try:
        to_be_aligned = directory + to_be_aligned
        template = directory + template

        assert isinstance(to_be_aligned,str)
        slash_index = to_be_aligned.rfind("/")
        fname = to_be_aligned[slash_index+1:]

        num_matches,image = align(to_be_aligned,template)

        if num_matches < 70000:
            print "bad match"
            with open("/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year+"/no_alignment.txt","a") as f:
                f.write(fname+"\n")
        else:
            cv2.imwrite("/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year+"/"+fname,image)
    except np.linalg.linalg.LinAlgError:
        print "linear algebra problem"
        with open("/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year+"/algebra_problem.txt","a") as f:
                f.write(to_be_aligned+"\n")