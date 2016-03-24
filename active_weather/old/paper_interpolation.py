from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
__author__ = 'ggdhines'


img = cv2.imread('/home/ggdhines/region.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
height,width,_ = img.shape
ret,thresh1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)

kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(gray,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

cv2.imwrite("/home/ggdhines/horizontal.jpg",close)


# mask.fill(0)
rows = []
# plt.imshow(img)
_,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    perimeter = cv2.arcLength(cnt,True)
    if (w/h > 5) and min(h,w) > 1 and (perimeter > 500):
        mask = np.zeros(close.shape,np.uint8)
        mask2 = np.zeros(close.shape,np.uint8)
        s = cnt.shape
        f = np.reshape(cnt,(s[0],s[2]))

        # print(f.shape)
        cv2.drawContours(mask,[cnt],0,255,-1)

        t = np.where(mask>0)
        pts = sorted(zip(t[1],t[0]),key = lambda x:x[0])



        # print(t)
        # assert False

        x,y = zip(*f)

        degrees = 2
        coeff = list(reversed(np.polyfit(x,y,degrees)))
        # print coeff
        #
        # plt.plot(x,y)
        y_min = min(y)
        y_max = max(y)
        # print y_min,y_max



        y_bar = [sum([coeff[p]*x_**p for p in range(degrees+1)]) for x_ in x]
        # plt.plot(x,y_bar)

        std = math.sqrt(np.mean([(y1-y2)**2 for (y1,y2) in zip(y,y_bar)]))
        print(std)

        def y_bar(x_,upper):
            return int(sum([coeff[p]*x_**p for p in range(degrees+1)]) + upper*std)

        # print f[:,0]
        domain = sorted(set(f[:,0]))
        y_vals = [y_bar(x,-1) for x in domain]
        y_vals.extend([y_bar(x,1) for x in list(reversed(domain))])
        x_vals = list(domain)
        x_vals.extend(list(reversed(domain)))
        pts = np.asarray(zip(x_vals,y_vals))

        cv2.drawContours(mask2,[pts],0,255,-1)

        # plt.plot(x_vals,y_vals)
        plt.imshow(mask2,cmap="gray")
        plt.show()

        overlap = np.min([mask,mask2],axis=0)

        _,contour, hier = cv2.findContours(overlap.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            s = cnt.shape
            f2 = np.reshape(cnt,(s[0],s[2]))

            plt.plot(f[:,0],f[:,1])
            plt.plot(f2[:,0],f2[:,1])
            # plt.plot(end_x,median_y)
            plt.ylim((y_max+10,y_min-10))
            plt.xlim((0,width))
            plt.show()
            assert False






