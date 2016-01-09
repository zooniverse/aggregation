#!/usr/bin/env python
__author__ = 'greghines'
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
db = client['chimp_2015-05-03']
subjects = db["chimp_subjects"]


def mse(imageA, imageB):
    # taken from
    # http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

true_positives = []
false_positives = []

# iterate over a set of subjects
# for each subject get the retirement reason - used to create gold standard data
for ii,s in enumerate(subjects.find().limit(250)):
    print ii
    id_ = s["zooniverse_id"]
    preview_url = s["location"]["previews"][0][0][:-5]
    reason = s["metadata"]["retire_reason"]

    # down every preview clip for this subject
    for i in range(1,16):
        url = preview_url + str(i) + ".jpg"
        fname = id_+"_"+str(i)+".jpg"
        if not(os.path.isfile(image_directory+fname)):
                urllib.urlretrieve(url, image_directory+fname)

    # find the maximum mse between all pairs of images
    differences = []
    for i in range(1,16):
        for j in range(i+1,16):
            fname1 = image_directory+id_+"_"+str(i)+".jpg"
            f1 = cv2.imread(fname1)
            fname2 = image_directory+id_+"_"+str(j)+".jpg"
            f2 = cv2.imread(fname2)

            f1 = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
            f2 = cv2.cvtColor(f2,cv2.COLOR_BGR2GRAY)
            differences.append(mse(f1,f2))

    # add the threshold value to either the false positive (if the movie was
    # classified as blank by users) or true positive
    if reason == "blank":
        false_positives.append(max(differences))
    else:
        true_positives.append(max(differences))

# create the ROC curve
alphas = true_positives[:]
alphas.extend(false_positives)
alphas.sort()
X = []
Y = []
for a in alphas:
    X.append(len([x for x in false_positives if x >= a])/float(len(false_positives)))
    Y.append(len([y for y in true_positives if y >= a])/float(len(true_positives)))

print len(false_positives)
plt.plot(X,Y)
plt.xlabel("False Positive Count")
plt.ylabel("True Positive Count")
plt.show()


