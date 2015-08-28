__author__ = 'greg'
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy
import cPickle
import os
from neural_network import Network,load_data_wrapper
# import cv2
from scipy import spatial


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

corner = 2858,1934

X_l2 = []
Y_l2 = []

pt_indices = tree.query_ball_point(corner,r=30)
for i in pt_indices:
    x = X_l[i]
    y = Y_l[i]
    X_l2.append(x)
    Y_l2.append(y)

    # plt.plot(x,y,"o",color="red")

plt.plot(corner[0],corner[1],"o",color="green")

# to_delete = set()
#
# threshold = 40
# r = 5
#
# for x,y in line_segment((586,1425),(780,1428)):
#     ball = tree.query_ball_point([x,y],r=r)
#     if len(ball) <= threshold:
#         for b in ball:
#             to_delete.add(b)
#     plt.plot(x,y,"o",color="red")
#
# for x,y in line_segment((780,1428),(883,1429)):
#     ball = tree.query_ball_point([x,y],r=r)
#     if len(ball) <= threshold:
#         for b in ball:
#             to_delete.add(b)
#     plt.plot(x,y,"o",color="red")
#
# to_delete = sorted(to_delete,reverse=True)
#
# plt.show()

# xy = zip(X_l,Y_l)
# for d in to_delete:
#     del xy[d]
#
# X_l,Y_l = zip(*xy)
# plt.plot(X_l,Y_l,"o",color="blue")
# plt.ylim((2000,1271))
# plt.xlim((585,3265))
# plt.show()
x_counter,x_bins= numpy.histogram(X_l2,len(X_l2)/10)
x_max = max(x_counter)
x1 = int(numpy.median([(x_bins[i]+x_bins[i+1])/2. for i in range(len(x_counter)) if x_counter[i] == x_max]))
# plt.show()
# n, bins, patches = plt.hist(Y_l2, len(Y_l2)/10, histtype='step')
y_counter,y_bins = numpy.histogram(Y_l2,len(Y_l2)/10)
y_max = max(y_counter)
y1= int(numpy.median([(y_bins[i]+y_bins[i+1])/2. for i in range(len(y_counter)) if y_counter[i] == y_max]))
plt.plot(x1,y1,"o",color="yellow")
plt.show()

