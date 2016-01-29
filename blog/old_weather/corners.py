import cv2
import numpy as np

img = cv2.imread("/home/ggdhines/Databases/old_weather/cells/Bear-AG-29-1939-0185_1_7.png")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,None)

cv2.imwrite('/home/ggdhines/sift_keypoints.jpg',img)