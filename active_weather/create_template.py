__author__ = 'ggdhines'
import numpy as np
import os
import cv2

image_so_far = None

ship = "Bear"
year = "1940"

aligned_images_dir = "/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year + "/"

aligned_images = list(os.listdir(aligned_images_dir))


# go through at most 100 images in the given directory
image_count = 0
for f_name in aligned_images[:5]:
    if f_name.endswith(".JPG"):
        if f_name in ["Bear-AG-29-1940-0022.JPG","Bear-AG-29-1940-0342.JPG"]:
            continue
        image = cv2.imread(aligned_images_dir+f_name,0)
        _,image = cv2.threshold(image,205,255,cv2.THRESH_BINARY)
        image = 255-image

        if image_so_far is None:
            image_so_far = image


            image_count += 1

        else:
            image_count += 1
            beta = 1/float(image_count)
            alpha = 1. - beta

            # image_so_far = cv2.addWeighted(image_so_far,alpha,image,beta,0)
            # image_so_far = np.asarray([image_so_far, image]).max(0)
            image_so_far = image_so_far & image

# kernel = np.ones((3,3),np.uint8)
# closing = cv2.morphologyEx(, cv2.MORPH_CLOSE, kernel)

# ret,thresh1 = cv2.threshold(image_so_far,180,255,cv2.THRESH_BINARY)
#
# thresh1 = 255 - image_so_far

cv2.imwrite("/home/ggdhines/invert_reference.jpeg",image_so_far)


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
