__author__ = 'greg'
import requests
import pymongo
import json
import csv
import os
import urllib
import numpy
import cv2
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt

from skimage.feature import CENSURE

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = numpy.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

token_mapping = {}

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

with open("/home/greg/Dropbox/cloudsight.csv","rb") as f:
    c = csv.reader(f)
    for url,token in c:
        token_mapping[url] = token

# connect to the mongo server
client = pymongo.MongoClient()
db = client['serengeti_2015-02-22']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

X_negative = []
X_positive = []

old_greys = None

for ii,subject in enumerate(subject_collection.find({"tutorial":{"$ne":True}}).limit(50000)):
    images = subject["location"]["standard"]
    if subject["coords"] != [-2.4672743413359295, 34.75278520232197]:
        continue

    retire_reason = subject["metadata"]["retire_reason"]
    if retire_reason == "blank_consensus":
        continue
    print retire_reason



    files = []

    for url in images:
        slash_index = url.rfind("/")
        fname = url[slash_index+1:]

        image_path = base_directory+"/Databases/images/"+fname

        files.append(image_path)

        if not(os.path.isfile(image_path)):
            print "downloading"
            urllib.urlretrieve(url, image_path)

    loaded_files = [cv2.imread(f) for f in files]
    grey_files = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in loaded_files]

    mse_values = []
    ssim_values = []

    detector = CENSURE()
    detector.detect(grey_files[0])
    kp0 = len(detector.keypoints)
    if len(grey_files) > 1:
        detector.detect(grey_files[1])
        kp1 = len(detector.keypoints)
    else:
        kp1 = None

    if len(grey_files) > 2:
        detector.detect(grey_files[2])
        kp2 = len(detector.keypoints)
    else:
        kp2 = None
    #print kp0,kp1,kp2
    print min([p for p in (kp0,kp1,kp2) if p is not None])
    continue

    if old_greys is not None:
        #
        # for i in range(len(loaded_files)-1):
        #     for j in range(i+1,len(loaded_files)):
        #         imageA = grey_files[i]
        #         imageB = grey_files[j]
        #
        #         mse_values.append(mse(imageA, imageB))
        #         ssim_values.append(ssim(imageA, imageB))
        for i in range(len(old_greys)):
            for j in range(len(grey_files)):
                imageA = old_greys[i]
                imageB = grey_files[j]

                # mse_values.append(mse(imageA, imageB))
                ssim_values.append(ssim(imageA, imageB))



    # print min(mse_values),numpy.mean(mse_values),numpy.median(mse_values),max(mse_values)
    # print min(ssim_values),numpy.mean(ssim_values),numpy.median(ssim_values),max(ssim_values)

        v = min(ssim_values)

        if retire_reason == "blank":
            X_positive.append(v)
        else:
            X_negative.append(v)

    old_greys = grey_files

alpha_list = X_negative[:]
alpha_list.extend(X_positive)
alpha_list.sort()

roc_X = []
roc_Y = []
for alpha in alpha_list:
    positive_count = sum([1 for x in X_positive if x >= alpha])
    positive_rate = positive_count/float(len(X_positive))

    negative_count = sum([1 for x in X_negative if x >= alpha])
    negative_rate = negative_count/float(len(X_negative))

    roc_X.append(negative_rate)
    roc_Y.append(positive_rate)



#print roc_X

plt.plot(roc_X,roc_Y,color="red")
plt.show()