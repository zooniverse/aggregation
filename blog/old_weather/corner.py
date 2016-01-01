__author__ = 'ggdhines'
import cv2
import numpy as np

filename = "/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG"
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,5,0.1)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imwrite("/home/ggdhines/t.png",img)