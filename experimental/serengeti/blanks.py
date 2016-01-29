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

# the directory to store the movie preview clips in
image_directory = "/home/greg/Databases/chimp/images/"

# connect to the mongodb server
client = pymongo.MongoClient()
db = client['b']
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
            urllib.urlretrieve(url, image_path)

print len(all_files)
for file_index in range(len(all_files)):
    good_count = 0
    img1 = cv2.imread("/home/ggdhines/Databases/images/"+all_files[file_index],0)          # queryImage
    for file2_index in range(len(all_files)):
        if file_index == file2_index:
            continue


        img2 = cv2.imread("/home/ggdhines/Databases/images/"+all_files[file2_index],0) # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SURF_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        print type(matches)
        # Apply ratio test
        good = []
        for m,n in matches:
            print m.distance,n.distance
            if m.distance < 0.2:
                good.append([m])

        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        print len(good)
        plt.imshow(img3),plt.show()

        if len(good) > 500:
            good_count += 1

    print "/home/ggdhines/Databases/images/"+all_files[file_index],good_count

