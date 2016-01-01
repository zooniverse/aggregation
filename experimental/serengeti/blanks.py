#!/usr/bin/env python
__author__ = 'greghines'
import matplotlib
matplotlib.use('WXAgg')
import numpy as np
import os
import pymongo
import urllib
import matplotlib.pyplot as plt
import cv2
import math
# the directory to store the movie preview clips in
image_directory = "/home/greg/Databases/chimp/images/"

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

# connect to the mongodb server
client = pymongo.MongoClient()
db = client['serengeti_2015-08-20']
subjects = db["serengeti_subjects"]

all_files = []

blanks = {}
images_to_subjects = {}

for ii,s in enumerate(subjects.find({"tutorial":{"$ne":True},"coords":[-2.4672743413359295, 34.75278520232197]}).limit(20)):
    print ii
    id_ = s["zooniverse_id"]
    blanks[id_] = []

    for url in s["location"]["standard"]:
        r_slash = url.rfind("/")
        fname = url[r_slash+1:]

        image_path = "/home/ggdhines/Databases/images/"+fname

        all_files.append(fname)

        images_to_subjects[fname] = id_

        if not(os.path.isfile(image_path)):
            prefix = "http://zooniverse-static.s3.amazonaws.com/"
            urllib.urlretrieve(prefix+url[7:], image_path)

# print len(all_files)
# for file_index in range(len(all_files)):
#     good_count = 0
#     img1 = cv2.imread("/home/ggdhines/Databases/images/"+all_files[file_index],0)          # queryImage
#     for file2_index in range(len(all_files)):
#         print file2_index
#         if file_index == file2_index:
#             continue
#
#
#         img2 = cv2.imread("/home/ggdhines/Databases/images/"+all_files[file2_index],0) # trainImage
#
#         warp_mode = cv2.MOTION_AFFINE
#         warp_matrix = np.eye(2, 3, dtype=np.float32)
#         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)
#         (cc, warp_matrix) = cv2.findTransformECC (get_gradient(img1), get_gradient(img2),warp_matrix, warp_mode, criteria)
#
#     print "/home/ggdhines/Databases/images/"+all_files[file_index],good_count

img = cv2.imread('simple.jpg',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# Print all default params
print "Threshold: ", fast.getInt('threshold')
print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
print "neighborhood: ", fast.getInt('type')
print "Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)