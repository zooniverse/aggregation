import matplotlib
matplotlib.use('WXAgg')
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pymongo
import os
import urllib
import Image
import ImageStat
import glob
import matplotlib.cbook as cbook

def brightness( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

# # connect to the mongodb server
# client = pymongo.MongoClient()
# db = client['b']
# subjects = db["serengeti_subjects"]
# classifications = db["serengeti_classifications"]
#
#
# base_subject = subjects.find_one({"coords":{"$ne":[]}})
# coords = base_subject["coords"]
#
# for subject in subjects.find({"coords":coords}).limit(100):
#     url = subject["location"]["standard"][0]
#
#     r_slash = url.rfind("/")
#     fname = url[r_slash+1:]
#
#     image_path = "/home/ggdhines/Databases/images/serengeti/"+fname
#
#     if not(os.path.isfile(image_path)):
#         urllib.urlretrieve(url, image_path)
#
#     print brightness(image_path)
#
# assert False

image_list = []
for fname in glob.glob("/home/ggdhines/Databases/images/time_series/*.jpg"):
    image_list.append(cv2.imread(fname,0))

base_image = np.median(image_list,axis=[0])
base_image = base_image.astype(np.uint8)
fig, ax1 = plt.subplots(1, 1)
ax1.imshow(base_image)
plt.show()

for image in image_list:
    diff = np.abs(base_image - image)

    avg = np.mean(diff)
    diff = np.abs(diff - avg)
    diff = diff.astype(np.uint8)

    ret,thresh1 = cv2.threshold(diff,127,255,cv2.THRESH_BINARY)

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(thresh1)
    plt.show()

    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(diff,-1,kernel)
    plt.imshow(dst)
    plt.show()


assert False

MIN_MATCH_COUNT = 10

# img1 = cv2.imread("/home/ggdhines/Databases/images/50c210188a607540b9000003_0.jpg",0)          # queryImage
f_names = ["/home/ggdhines/Databases/images/50c210188a607540b900000e_1.jpg","/home/ggdhines/Databases/images/50c210188a607540b900000f_0.jpg","/home/ggdhines/Databases/images/50c210188a607540b900000e_0.jpg","/home/ggdhines/Databases/images/50c210188a607540b9000011_0.jpg","/home/ggdhines/Databases/images/50c210188a607540b900000c_1.jpg","/home/ggdhines/Databases/images/50c210188a607540b9000014_1.jpg","/home/ggdhines/Databases/images/50c210188a607540b9000012_1.jpg"]

# image = plt.imread(f_names[0])

# fig, ax1 = plt.subplots(1, 1)
# ax1.imshow(image)
# temp_image = cv2.imread(f_names[0])
# blue = temp_image[:,:,2] > 100
# y,x = np.where(blue)
# plt.plot(x,y,".")
# plt.ylim((temp_image.shape[0],0))
# plt.xlim((0,temp_image.shape[1]))
# plt.show()

# images = []
for f in f_names:
    temp_image = cv2.imread(f)
    R = temp_image[:,:,0]
    G = temp_image[:,:,1]
    B = temp_image[:,:,2]

    bool1 = abs(R-G) < 5
    bool2 = abs(G-B) < 5
    bool3 = B>R
    bool4 = B>G
    bool5 = B > 50
    bool6 = B < 230
    sky = bool1 & bool2 & bool3 & bool4 & bool5 & bool6

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(temp_image)
    # temp_image = cv2.imread(f_names[0])

    y,x = np.where(sky)
    print y,x
    plt.plot(x,y,".")
    # plt.ylim((temp_image.shape[0],0))
    # plt.xlim((0,temp_image.shape[1]))
    plt.show()

    # assert False
    # images.append(cv2.imread(f))

# kernel = np.ones((5,5),np.float32)/25
# img2 = cv2.filter2D(img2,-1,kernel)
# img3 = cv2.filter2D(img3,-1,kernel)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img3 = clahe.apply(img3)
# img2 = clahe.apply(img2)

# diff = cv2.subtract(img2,img3)
# cv2.imwrite('/home/ggdhines/diff1.jpg',diff)
#
# diff = cv2.subtract(img3,img2)
# cv2.imwrite('/home/ggdhines/diff2.jpg',diff)
#
# diff2 = (img2+img3)/2
# cv2.imwrite('/home/ggdhines/diff3.jpg',diff2)

# img3 = clahe.apply(img3)
# img4 = clahe.apply(img4)
# img5 = clahe.apply(img5)
# img6 = clahe.apply(img6)
# img7 = clahe.apply(img7)
#


# dst_2 = cv2.filter2D(diff2,-1,kernel)
# cv2.imwrite('/home/ggdhines/diff4.jpg',dst_2)
# dst_3 = cv2.filter2D(img3,-1,kernel)
# dst_4 = cv2.filter2D(img4,-1,kernel)

# base_image = (dst+dst_2+dst_3+dst_4)/4
# a = np.asarray([img1,img2,img3,img4,img5,img6,img7])
a = np.asarray(images)
print a.shape
base_image = np.median(a,axis=[0])
# base_image = cv2.filter2D(base_image,-1,kernel)
# base_image = base_image.astype(np.uint8)
print type(base_image[0][0])
print base_image

fig, ax1 = plt.subplots(1, 1)
ax1.imshow(base_image)
# temp_image = cv2.imread(f_names[0])
blue = base_image[:,:,2] > 150
y,x = np.where(blue)
plt.plot(x,y,".")
# plt.ylim((temp_image.shape[0],0))
# plt.xlim((0,temp_image.shape[1]))
plt.show()

# img5 = cv2.subtract(base_image,dst)
cv2.imwrite('/home/ggdhines/base_image.jpg',base_image)
blurred = cv2.blur(base_image,(5,5))
cv2.imwrite('/home/ggdhines/blurred_base.jpg',blurred)
# test_image = cv2.filter2D(img2,-1,kernel)


# print test_image
# test_image = test_image.astype(float)
# img1 = cv2.blur(images[4],(5,5))
diff = np.absolute(images[3]-base_image)
diff = diff.astype(np.uint8)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

# assert False
#
# print type(img4[0][0])
# diff = cv2.subtract(test_image,base_image)
# diff2 = cv2.subtract(base_image,test_image)
# diff = clahe.apply(diff)
cv2.imwrite('/home/ggdhines/diff1.jpg',opening)
# cv2.imwrite('/home/ggdhines/clahe_6.jpg',diff2)
# cv2.imwrite('/home/ggdhines/clahe_5.jpg',test_image)
assert False
# # create a CLAHE object (Arguments are optional).

#

img3 = clahe.apply(img3)
img4 = clahe.apply(img4)
# cv2.imwrite('/home/ggdhines/clahe_1.jpg',cl1)
#
# cl2 = clahe.apply(img2)
# cv2.imwrite('/home/ggdhines/clahe_2.jpg',cl2)
#
# cl3 = clahe.apply(img3)
# cv2.imwrite('/home/ggdhines/clahe_2.jpg',cl2)
#
# cl4 = (cl1 + cl2)/2
# kernel = np.ones((10,10),np.float32)/25
# dst = cv2.filter2D(cl4,-1,kernel)
# cl5 = cl3 - dst

print img1
print
print img2

img5 = (img1+img2+img3+img4)/4
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img5, cv2.MORPH_CLOSE, kernel)
print img3
print
# img4 = img3 - img2
img5 = cv2.subtract(img3,img4)
print img4
cv2.imwrite('/home/ggdhines/clahe_3.jpg',opening)
assert False

# img = cv2.imread('wiki.jpg',0)
equ2 = cv2.equalizeHist(img2)
res = np.hstack((img2,equ2)) #stacking images side-by-side
cv2.imwrite('/home/ggdhines/res.png',res)

# img = cv2.imread('wiki.jpg',0)
equ1 = cv2.equalizeHist(img1)
res = np.hstack((img1,equ1)) #stacking images side-by-side
cv2.imwrite('/home/ggdhines/res1.png',res)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
kp = sift.detect(img2,None)
img=cv2.drawKeypoints(img2,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,outImage=None)
cv2.imwrite('/home/ggdhines/sift_keypoints2.jpg',img)
# assert False

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(equ1,None)
kp2, des2 = sift.detectAndCompute(equ2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()