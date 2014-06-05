#!/usr/bin/env python
from __future__ import print_function
import pymongo
import os
from subprocess import call
from PIL import Image
import numpy as np
import ephem
__author__ = 'ggdhines'

import scipy as sp
from scipy.misc import imread
from scipy.signal.signaltools import correlate2d as c2d

from scipy.linalg import norm
from scipy import sum, average

size = 128, 128

os.chdir("/home/ggdhines/Databases/serengeti/photos/")

client = pymongo.MongoClient()
db = client['serengeti_2014-05-13']
collection = db['serengeti_subjects']

blank_photos = []
photo_names = []
daylight_photos = []

o=ephem.Observer()
o.lat='-2.4672743413359295'
o.lon='34.75278520232197'

def get(fName):
    # get JPG image as Scipy array, RGB (3 layer)
    data = imread(fName)
    # convert to grey-scale using W3C luminance calc
    data = sp.inner(data, [299, 587, 114]) / 1000.0
    # normalize per http://en.wikipedia.org/wiki/Cross-correlation
    return (data - data.mean()) / data.std()


def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

blankFiles = []
animalFiles = []

for document in collection.find({"coords": [-2.4672743413359295, 34.75278520232197]})[0:200]:
    photo_list = document["location"]["standard"]


    firstPhoto = photo_list[0]

    i = firstPhoto.rfind("/")
    i2 = firstPhoto.rfind(".")
    image_id = str(firstPhoto[i+1:i2])
    p_l = [image_id]

    if not(os.path.isfile("day/"+image_id+".thumbnail")):
        continue

    #now if necessary, download any additional photos
    for photo in photo_list[1:]:
        i = photo.rfind("/")
        i2 = photo.rfind(".")
        image_id = str(photo[i+1:i2])

        p_l.append(image_id)

        #download and convert to thumbnail
        if not(os.path.isfile("day/"+image_id+".thumbnail")):
            try:
                call(["cp",image_id+".thumbnail","day/"])
            except ValueError:
                call(["aws", "s3", "cp", "s3://www.snapshotserengeti.org/subjects/standard/"+image_id+".jpg", "."])
                im = Image.open(image_id+".jpg")
                im.thumbnail(size, Image.ANTIALIAS)
                im.save(image_id + ".thumbnail", "JPEG")


    if document["metadata"]["retire_reason"] == "blank":
        blank_photos.append(True)
        blankFiles.append(p_l[:])
    else:
        blank_photos.append(False)
        animalFiles.append(p_l[:])

    daylight_photos.append(p_l[:])

blankDiff = []
blankIndex = []
animalDiff = []
animalIndex = []

for i in range(len(daylight_photos)):
    print(i)
    #check first for movement
    photos = daylight_photos[i]
    maxDiff = 0
    for pIndex,p1 in enumerate(photos[:-1]):
        for p2 in photos[pIndex+1:]:
            img1 = to_grayscale(imread("day/"+p1+".thumbnail").astype(float))
            img2 = to_grayscale(imread("day/"+p2+".thumbnail").astype(float))

            n_m, n_0 = compare_images(img1, img2)
            diff = n_m/img1.size
            maxDiff = max(maxDiff, diff)

    #cutoff will be 9 (for now - just picking a value)
    if maxDiff >= 9.:
        #assume that there is an animal there
        continue

    #similarity test
    similarity_l = []
    for blank_pic,possible_pic in zip(blankFiles[0],photos):
        p1 = get("day/"+blank_pic+".thumbnail")
        p2 = get("day/"+possible_pic+".thumbnail")

        similarity_l.append(c2d(p1, p2, mode='same').max())

    overallSimilarity = min(similarity_l)

    if photos in blankFiles:
        blankDiff.append(overallSimilarity)
        blankIndex.append(i)
    else:
        animalDiff.append(overallSimilarity)
        animalIndex.append(i)

        if (i == 7):
            print(overallSimilarity)
            print(daylight_photos[i])
            assert(False)


#img1 = to_grayscale(imread("day/"+blankFiles[0]+".thumbnail").astype(float))

#similarityList = []

# for i in range(1,len(blankFiles)):
#     print(i)
#     #img2 = to_grayscale(imread("day/"+blankFiles[i]+".thumbnail").astype(float))
#     s = []
#     for p1 in blankFiles[0]:
#         for p2 in blankFiles[i]:
#             img1 = to_grayscale(imread("day/"+p1+".thumbnail").astype(float))
#             img2 = to_grayscale(imread("day/"+p2+".thumbnail").astype(float))
#
#             s.append(c2d(img1, img2, mode='same').max())
#
#     similarityList.append(np.mean(s))
#
print(max(blankDiff))
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(blankIndex, blankDiff, 'o', color="green")
ax.plot(animalIndex, animalDiff, 'o', color="red")
plt.show()

#base1 = get("day/50c212448a607540b901d504_0.thumbnail")
#base2 = get("day/50c212438a607540b901d4a0_0.thumbnail")
#img1 = to_grayscale(imread("day/50c212438a607540b901d49d_0.thumbnail").astype(float))


# for i in range(0,len(daylight_photos)):
#     print(i)
#     overallSimilarity = None
#     for i2,image_id in enumerate(daylight_photos[i][:-1]):
#         for image_id2 in daylight_photos[i][i2+1:]:
#             img1 = to_grayscale(imread("day/"+image_id+".thumbnail").astype(float))
#             img2 = to_grayscale(imread("day/"+image_id2+".thumbnail").astype(float))
#
#             n_m, n_0 = compare_images(img1, img2)
#             similarity = n_m/img1.size
#         #similarity = c2d(base1, im2, mode='same').max()
#             if (overallSimilarity is None) or (similarity < overallSimilarity):
#                 overallSimilarity = similarity
#     #similarity2 = c2d(base2, im2, mode='same').max()
#     #similarity = (similarity1+similarity2)/2.
#     #similarity = max(similarity1,similarity2)
#     #print(similarity)
#
#
#     # img2 = to_grayscale(imread("day/"+daylight_photos[i]+".thumbnail").astype(float))
#     # try:
#     #     n_m, n_0 = compare_images(img1, img2)
#     # except ValueError:
#     #     print("====---")
#     #     continue
#     #
#     # diff = n_m/img1.size
#
#     if blank_photos[i]:
#         blankX.append(i)
#         blankY.append(overallSimilarity)
#     else:
#         animalsX.append(i)
#         animalsY.append(overallSimilarity)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(blankX, blankY, 'o', color="green")
# ax.plot(animalsX, animalsY, 'o', color="red")
# plt.show()
#
# f = sorted(zip(animalFiles,animalsY),key = lambda x:x[1])
# print(f[-1])