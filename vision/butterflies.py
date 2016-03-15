from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
from aggregation_api import AggregationAPI
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import cv2
import numpy as np

butterflies = AggregationAPI(1150,"development")
butterflies.__setup__()

# fname = butterflies.__image_setup__(1120709)[0]
fname = butterflies.__image_setup__(1500825)[0]

image_file = cbook.get_sample_data(fname)
image = plt.imread(image_file)

res = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
# dy = cv2.Sobel(res,cv2.CV_16S,0,2)
# dy = cv2.convertScaleAbs(dy)
# cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
# ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# t= image
# _,contour, hier = cv2.findContours(close.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contour:
#     x,y,w,h = cv2.boundingRect(cnt)
#     if ((w/h)>5) and (w>130) and (w < 160):
#         print(w)
#         cv2.drawContours(t,[cnt],0,(0,255,0),-1)
#
# im = plt.imshow(t)
# plt.show()


edges = cv2.Canny(res,50,100)
plt.imshow(edges,cmap="gray")
plt.show()

th3 = cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,351,2)
plt.imshow(th3,cmap="gray")
plt.show()

ret2,th2 = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(th2,cmap="gray")
plt.show()

_,contour, hier = cv2.findContours(res.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
t = np.zeros(res.shape,np.uint8)
for cnt,h in zip(contour,hier):
    print(h)
    cv2.drawContours(t,[cnt],0,255,-1)

plt.imshow(t,cmap="gray")
plt.show()
#     x,y,w,h = cv2.boundingRect(cnt)