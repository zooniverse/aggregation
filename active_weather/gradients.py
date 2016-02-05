import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/ggdhines/Databases/old_weather/pruned_cases/Bear-AG-29-1939-0185.JPG',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=7)
#
#
fig, ax1 = plt.subplots(1, 1,figsize=(53, 78), dpi=72)
ax1.imshow(img,cmap = 'gray')
# plt.imshow()

# plt.imshow(sobely,cmap = 'gray')
# plt.show()

kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
dy = cv2.Sobel(img,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

_,contours, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for ii,cnt in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt)

    if w/h > 5:
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        s = approx.shape
        s2 = cnt.shape
        approx = approx.reshape((s[0],s[2]))
        cnt = cnt.reshape((s2[0],s2[2]))
        x,y = zip(*approx)

        x = list(x)
        y = list(y)

        x.append(x[0])
        y.append(y[0])

        ax1.plot(x,y)

        # rect = cv2.minAreaRect(approx)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # print box
        im = cv2.drawContours(close,contours,ii,(0,0,255),2)
        # im = cv2.drawContours(close,[approx],0,(0,0,255),2)

        # cv2.drawContours(close,[approx],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()
plt.savefig("/home/ggdhines/weather.jpg", bbox_inches='tight')
# plt.imshow(closey,cmap = 'gray')
# plt.show()

# kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))
# dy = cv2.Sobel(img,cv2.CV_16S,0,2)
# dy = cv2.convertScaleAbs(dy)
# cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
# ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)
#
# _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contour:
#     x,y,w,h = cv2.boundingRect(cnt)
#     if w/h > 5:
#         cv2.drawContours(close,[cnt],0,255,-1)
#     else:
#         cv2.drawContours(close,[cnt],0,0,-1)
#
# close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
# closey = close.copy()
#
# plt.imshow(closey,cmap = 'gray')
# plt.show()