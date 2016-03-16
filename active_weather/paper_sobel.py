import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('/home/ggdhines/1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(gray,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

cv2.imwrite("/home/ggdhines/horizontal.jpg",close)

t = np.zeros(close.shape,np.uint8)
_,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(t,[cnt],0,255,1)

plt.imshow(t,cmap="gray")
plt.show()

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()
