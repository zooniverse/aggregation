import cv2
import numpy as np

template = np.zeros((500,500,3),np.uint8)
template[:,:,0] = 255
template[:,:,1] = 255
template[:,:,2] = 255

x = [50,250,250,50,50]
y = [50,50,250,250,50]
cnt = np.asarray(zip(x,y))
cv2.drawContours(template,[cnt],0,0,1)



x = [100,200,200,100,100]
y = [300,300,150,150,300]
cnt = np.asarray(zip(x,y))
cv2.drawContours(template,[cnt],0,0,1)

x = [150,400,400,150,150]
y = [200,200,400,400,200]
cnt = np.asarray(zip(x,y))
cv2.drawContours(template,[cnt],0,0,1)

x = [150,200,200,150,150]
y = [250,250,200,200,250]
cnt = np.asarray(zip(x,y))
cv2.drawContours(template,[cnt],0,(255,0,0),-1)


cv2.imwrite("/home/ggdhines/github/aggregation/docs/images/rectangle_overlap.jpg",template)