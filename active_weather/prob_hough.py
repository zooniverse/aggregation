import matplotlib
matplotlib.use('WXAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/ggdhines/1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 50
maxLineGap = 25
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for l in lines:
    x1,y1,x2,y2 = l[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('/home/ggdhines/2.jpg',img)