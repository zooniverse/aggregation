__author__ = 'ggdhines'
import cv2
import numpy as np
import glob
import random
# image1 = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0023.JPG",0)
# image2 = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0136.JPG",0)


f_names = ["/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0023.JPG","/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0136.JPG","/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0159.JPG"]
f_names.append("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0315.JPG")
image_so_far = None

ship = "Bear"
year = "1940"

aligned_images_dir = "/home/ggdhines/Databases/old_weather/aligned_images/"+ship+"/"+year + "/"

files = list(glob.glob(aligned_images_dir+"*.JPG"))

random.shuffle(files)

def read_file(fname):
    image = cv2.imread(fname,0)


    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # cv2.imwrite("/home/ggdhines/temp.jpg",image)
    # assert False


    # _,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY)

    # image = 255 - image
    # image = image > 0
    image = image.astype(np.float)

    return image

temp = read_file(files[0])
base = np.zeros(temp.shape)

l_ = 40

a = []

for i in range(l_):#range(len(files)/2):
    a.append(read_file(files[i]))
    continue

    # print i
    # f1 = read_file(files[2*i])
    # f2 = read_file(files[2*i+1])
    # # f3 = read_file(files[3*i+2])
    #
    # # c = np.max([f1,f2],axis=0)
    #
    # c = (f1+f2)/2.
    # c = c.astype(np.uint8)
    # cv2.imwrite("/home/ggdhines/temp.jpg",c)
    # assert False
    #
    # # xy = np.where(c)
    # #
    # # t = np.zeros(f1.shape)
    # # t[xy] = 255
    # # t = 255 - t
    # #
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',c)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # base += c

t = np.sum(a,axis=0)
t /= 255.
t = t.astype(np.uint8)
cv2.imwrite("/home/ggdhines/test.jpg",t)

template = np.percentile(a,50,axis=0)
cv2.imwrite("/home/ggdhines/temp.jpg",template)
template = 255 - template

sobely = cv2.Sobel(template,cv2.CV_64F,0,1,ksize=5)
sobely = sobely.astype(np.uint8)
shape = sobely.shape
_,contour, hier = cv2.findContours(sobely.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
horizontal_lines = np.zeros(shape,np.uint8)

for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(horizontal_lines,[cnt],0,255,-1)

cv2.imwrite("/home/ggdhines/horizontal.jpg",horizontal_lines)

sobely = cv2.Sobel(template,cv2.CV_64F,0,1,ksize=5)
sobely = sobely.astype(np.uint8)
shape = sobely.shape
_,contour, hier = cv2.findContours(sobely.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
horizontal_lines = np.zeros(shape,np.uint8)

for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(horizontal_lines,[cnt],0,255,-1)

cv2.imwrite("/home/ggdhines/horizontal.jpg",horizontal_lines)

# vertical

kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

dx = cv2.Sobel(template,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

_,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()
cv2.imwrite("/home/ggdhines/vertical.jpg",closex)

# base /= float(l_)
# t = base.astype(np.uint8)
# print image1.shape
#
#
#
# image2 = cv2.adaptiveThreshold(image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#
# image1 = 255 - image1
# image2 = 255 - image2
# image1 = image1 > 0
# image2 = image2 > 0
#
# bool_image = image1 & image2
# xy = np.where(image_so_far)
#
# t = np.zeros(image_so_far.shape)
# t[xy] = 255
# t = 255 - t
#
# _,t = cv2.threshold(t,180,255,cv2.THRESH_BINARY)
# cv2.imwrite("/home/ggdhines/text.jpg",t)