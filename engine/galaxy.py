__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.cluster import DBSCAN
import numpy as np
import math

# project = AggregationAPI(1138,"development")
# project.__image_setup__(1107600)

image_file = cbook.get_sample_data("/home/ggdhines/Databases/images/1b693d3b-50d0-4b1e-aa33-80115e8bcb81.jpeg")
image = plt.imread(image_file)

pts_x = []
pts_y = []

refer = [135, 111 , 65]
# image = rgb2gray(image)

# plt.close()
# fig = plt.figure()
# axes = fig.add_subplot(1, 1, 1)
# im = axes.imshow(image)
# plt.show()


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        dist = math.sqrt(sum([(a-b)**2 for (a,b) in zip(refer,image[i][j])]))
        # print dist
        if dist < 100:
            # plt.plot(i,j,color="blue")
            pts_x.append(i)
            pts_y.append(j)
print "here"
plt.plot(pts_y,pts_x,".",color="blue")
plt.xlim((0,512))
plt.ylim((512,0))
plt.show()





