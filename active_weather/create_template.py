__author__ = 'ggdhines'
import numpy as np
import os
import cv2
from copy import deepcopy



ship = "Bear"
year = "1940"

aligned_images_dir = "/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year + "/"

aligned_images = list(os.listdir(aligned_images_dir))


def blend_images(images,to_exclude = []):
    # go through at most 100 images in the given directory
    image_count = 0
    image_so_far = None
    for f_name in images:
        if f_name in to_exclude:
            continue

        if f_name.endswith(".JPG"):
            # if f_name in ["Bear-AG-29-1940-0022.JPG","Bear-AG-29-1940-0342.JPG"]:
            #     continue
            image = cv2.imread(aligned_images_dir+f_name,0)
            # _,image = cv2.threshold(image,205,255,cv2.THRESH_BINARY)
            # image = 255-image

            if image_so_far is None:
                image_so_far = image


                image_count += 1

            else:
                image_count += 1
                beta = 1/float(image_count)
                alpha = 1. - beta

                image_so_far = cv2.addWeighted(image_so_far,alpha,image,beta,0)
                # image_so_far = np.asarray([image_so_far, image]).max(0)
                # image_so_far = image_so_far & image

    # kernel = np.ones((3,3),np.uint8)
    # closing = cv2.morphologyEx(, cv2.MORPH_CLOSE, kernel)

    ret,thresh1 = cv2.threshold(image_so_far,180,255,cv2.THRESH_BINARY)
    thresh1 = 255-thresh1

    return thresh1

starting_template = blend_images(aligned_images)

# thresh1 = 255 - image_so_far
# thresh = cv2.adaptiveThreshold(image_so_far,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,0)
cv2.imwrite("/home/ggdhines/starting_template.jpeg",starting_template)

im2, contours, hierarchy = cv2.findContours(starting_template.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
# cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

template = np.zeros((starting_template.shape[0],starting_template.shape[1],3))

hierarchy = hierarchy.reshape((hierarchy.shape[1],hierarchy.shape[2]))
mask = np.zeros((starting_template.shape),np.uint8)

for cnt,h in zip(contours,hierarchy):
    next_,prev_,child_,parent_ = h
    if parent_ == 0:
        cv2.drawContours(template, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(mask,[cnt],0,255,-1)
        cv2.drawContours(mask,[cnt],0,0,2)


constrained_template = cv2.bitwise_and(starting_template,mask)
cv2.imwrite("/home/ggdhines/with_contours.jpeg",template)
cv2.imwrite("/home/ggdhines/mask.jpeg",mask)
cv2.imwrite("/home/ggdhines/constrained_template.jpeg",constrained_template)

t = constrained_template > 0
total = len(np.where(t)[0])

bad_matches = []

for f_name in aligned_images:
    if f_name.endswith(".JPG"):

        image = cv2.imread(aligned_images_dir+f_name,0)
        ret,image = cv2.threshold(image,190,255,cv2.THRESH_BINARY)
        # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

        image = 255 - image

        overlap = constrained_template & image
        overlap_count = len(np.where(overlap>0)[0])

        # cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # cv2.namedWindow("image2",cv2.WINDOW_NORMAL)
        # cv2.imshow('image2',overlap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        percent = overlap_count/float(total)

        if percent < 0.2:
            bad_matches.append(f_name)
        print f_name + "\t" + str(percent)
assert False
updated_template = blend_images(aligned_images,bad_matches)
cv2.imwrite("/home/ggdhines/updated_template.jpeg",updated_template)
assert False

_,contour, hier = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros((image_so_far.shape),np.uint8)

for cnt in contour:
    # x,y,w,h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)

    if area < 1000:

        cv2.drawContours(thresh1,[cnt],0,0,2)
    else:
        print area
        cv2.drawContours(thresh1,[cnt],0,255,2)

cv2.imwrite("/home/ggdhines/cleaned.jpeg",thresh1)


kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernelx)
cv2.imwrite("/home/ggdhines/kerneled.jpeg",closing)

# cv2.imwrite("/home/ggdhines/average_image.jpeg",image_so_far)
#
#
#
# cv2.imwrite("/home/ggdhines/thres.jpeg",thresh1)
#
# # cv2.drawContours(thresh1,contour,-1,(0,0,255),3)
#
# # cv2.imwrite("/home/ggdhines/base.jpeg",thresh1)
#
#
#
# # cv2.imwrite("/home/ggdhines/template.jpeg",thresh1)
#
# # template_size = len(np.where(thresh1>0)[0])
# #
# # for f_name in aligned_images:
# #     if f_name.endswith(".JPG"):
# #         image = cv2.imread(aligned_images_dir+f_name,0)
# #         _,thresh2 = cv2.threshold(image,180,255,cv2.THRESH_BINARY)
# #         matches = thresh2&thresh1
# #
# #         num_matches = len(np.where(matches)[0])
# #         print f_name + "\t" + str(num_matches/float(template_size))
#
# kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
# closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernelx)
#
# cv2.imwrite("/home/ggdhines/thres.jpeg",closing)

# dx = cv2.Sobel(thresh1,cv2.CV_16S,1,0)
# dx = cv2.convertScaleAbs(dx)
# cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
# _,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

# _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contour:
#     x,y,w,h = cv2.boundingRect(cnt)
#
#     if cv2.arcLength(cnt,True) < 100:
#         continue
#
#     if h/w > 5:
#         cv2.drawContours(close,[cnt],0,255,-1)
#     else:
#         cv2.drawContours(close,[cnt],0,0,-1)
# cv2.imwrite("/home/ggdhines/vert_lines.jpeg",close)

# sobelx = cv2.Sobel(thresh1,cv2.CV_64F,0,1,ksize=5)
# sobelx = sobelx.astype(np.uint8)
#
# _,contour, hier = cv2.findContours(sobelx,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
# base = np.zeros(image_so_far.shape)
#
# for cnt in contour:
#     x,y,w,h = cv2.boundingRect(cnt)
#
#     if cv2.arcLength(cnt,True) < 300:
#         continue
#
#     print cv2.arcLength(cnt,True)
#
#     if w/h > 5:
#         cv2.drawContours(base,[cnt],0,255,-1)
#
# cv2.imwrite("/home/ggdhines/vert_lines.jpeg",base)
#
#
# cv2.imwrite("/home/ggdhines/vert_lines2.jpeg",sobelx)
#
# ####
#
# # kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
# #
# # dx = cv2.Sobel(thresh1,cv2.CV_16S,0,1)
# # dx = cv2.convertScaleAbs(dx)
# # cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
# # _,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
# #
# # _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # for cnt in contour:
# #     x,y,w,h = cv2.boundingRect(cnt)
# #
# #     if cv2.arcLength(cnt,True) < 100:
# #         continue
# #
# #     if h/w > 5:
# #         cv2.drawContours(close,[cnt],0,255,-1)
# #     else:
# #         cv2.drawContours(close,[cnt],0,0,-1)
# # cv2.imwrite("/home/ggdhines/vert_lines.jpeg",close)
# #
# # # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
