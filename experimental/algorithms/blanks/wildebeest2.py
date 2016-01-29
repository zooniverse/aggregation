import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread("/home/greg/widlebeest/e614851a-17aa-4499-b410-f2e9985d5e1c.jpeg")
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

X = []
Y = []

img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
for i in range(len(img)):
    for j in range(len(img[0])):
        if min(img[i][j])>0:
            X.append(-i)
            Y.append(j)

plt.plot(Y,X,'.')
plt.show()