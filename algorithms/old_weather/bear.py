__author__ = 'greg'
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy
import cPickle
import os
from neural_network import Network,load_data_wrapper
import cv2
from scipy import spatial

start_x = 585.5
end_x = 3265.5
horizontal_lines = [(1271.5,1288.5),(1368.5,1384),(1424,1440),(1481.5,1497),(1537.5,1552),(1593,1607.5),(1649,1663),(1705.5,1718.5),(1762.5,1775.5),(1819,1830.5),(1875,1886),(1931,1942),(1987,1998)]

vertical_lines = [586,778.5,882,948,1049.5,1049.5,1117,1293,1456.5,1557.5,1758.5,1852,1959,2058.5,2157.5,3225,2559.5,2731,2855.5,2959,3095,3266]
start_y = 1271.5
end_y = 1998

def intersection(((y1,y2),(x1,x2)),x_):
    m = (y2-y1)/float(x2-x1)
    b = y2 - m*x2

    y_intersect = m*x_+b

    return x_,y_intersect

def line_segment((x1,y1),(x2,y2)):
    assert x1 < x2
    m = (y2-y1)/float(x2-x1)
    b = y1 - m*x1

    for x_i in range(int(x1),int(x2)):
        y_i = m*x_i+b
        yield (x_i,y_i)

    raise StopIteration()

image_file = cbook.get_sample_data("/home/greg/Databases/old_weather/images/Bear-AG-29-1939-0173.JPG")
image = plt.imread(image_file)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
im = axes.imshow(image)

for y_l in horizontal_lines:
    for x in vertical_lines:
        x_intersect,y_intersect = intersection((y_l,(start_x,end_x)),x)
        plt.plot(x_intersect,y_intersect,"o",color="red")

plt.show()

ref1 = [218,219,213]

X_l = []
Y_l = []

for y in range(1271,2000):
    # print x
    # x_temp = image[y]
    for x in range(585,3265):
        dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(image[y][x],ref1)]))

        if dist1 > 25:
            X_l.append(x)
            Y_l.append(y)
plt.plot(X_l,Y_l,"o",color="blue")

plt.ylim((2000,1271))
plt.xlim((585,3265))


tree = spatial.KDTree(zip(X_l,Y_l))

# corner = 2858,1934



def corner_correction(corner):
    X_l2 = []
    Y_l2 = []

    pt_indices = tree.query_ball_point(corner,r=30)
    for i in pt_indices:
        x = X_l[i]
        y = Y_l[i]
        X_l2.append(x)
        Y_l2.append(y)

        plt.plot(x,y,"o",color="red")

    # plt.plot(corner[0],corner[1],"o",color="green")

    x_counter,x_bins= numpy.histogram(X_l2,len(X_l2)/10)
    x_max = max(x_counter)
    x1 = int(numpy.median([(x_bins[i]+x_bins[i+1])/2. for i in range(len(x_counter)) if x_counter[i] == x_max]))
    # plt.show()
    # n, bins, patches = plt.hist(Y_l2, len(Y_l2)/10, histtype='step')
    y_counter,y_bins = numpy.histogram(Y_l2,len(Y_l2)/10)
    y_max = max(y_counter)
    y1= int(numpy.median([(y_bins[i]+y_bins[i+1])/2. for i in range(len(y_counter)) if y_counter[i] == y_max]))

    return x1,y1

tl_x,tl_y = corner_correction((1118,1593))
# plt.plot(1118,1593,"o",color="green")
# plt.plot(x1,y1,"o",color="yellow")
bl_x,bl_y = corner_correction((1118,1650))
# plt.plot(1118,1650,"o",color="green")
# plt.plot(x1,y1,"o",color="yellow")
tr_x,tr_y = corner_correction((1294,1593))
# plt.plot(1294,1593,"o",color="green")
# plt.plot(x1,y1,"o",color="yellow")
br_x,br_y = corner_correction((1294,1650))
# plt.plot(1294,1650,"o",color="green")
# plt.plot(x1,y1,"o",color="yellow")

# print image

# s = image.shape[0]*image.shape[1]
# print s
# print image.shape[0],image.shape[1]
# image.reshape((s,1,3))
#
#
# print image[0:10]


rows = numpy.array([[tl_y+i for j in range(tr_x-tl_x+1)] for i in range(bl_y-tl_y+1)])
print rows
columns = numpy.array([[tl_x+j for j in range(tr_x-tl_x+1)] for i in range(bl_y-tl_y+1)])
print columns
subimage = image[rows,columns]

for i in range(subimage.shape[0]):
    for j in range(subimage.shape[1]):
        dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(subimage[i][j],ref1)]))
        if dist1 < 25:
            subimage[i][j] = [255,255,255]

cv2.imwrite("/home/greg/output.png", image[rows,columns])
# plt.show()

