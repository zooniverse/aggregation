__author__ = 'greg'
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.datasets import SupervisedDataSet
# from pybrain.supervised.trainers import BackpropTrainer
import cPickle
import gzip

# Third-party libraries
import numpy as np


# net = buildNetwork(784, 30, 10)
# ds = SupervisedDataSet(784, 10)
#

#
#     ds.addSample(i,j)
#
# trainer = BackpropTrainer(net, ds)
# trainer.trainUntilConvergence()
#
# assert False

# X = numpy.asarray([285,319,354,422,745,779,871,928,963,990,1034,1068,1160,1228,1262])
# Y = numpy.asarray([272,339,369,398,428,457,488,515,545,573,602,631,660])

X = [285,319,354,422,745,779,871,928,963,990,1034,1068,1160,1228,1262]
Y = [272,339,369,398,428,457,488,515,545,573,602,631,660]

# @jit
def s():
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)

    image_file = cbook.get_sample_data("/home/greg/Dropbox/vol072_237_0.png")
    image = plt.imread(image_file)
    r = []

    # fig, ax = plt.subplots()
    # im = axes.imshow(image)


    #
    # for y in Y:
    #     plt.plot([X[0],X[-1]],[y,y],color="blue")
    #
    # for x in X:
    #     plt.plot([x,x],[Y[0],Y[-1]],color="blue")

    # for x_index in range(len(X)-1):
    #     for y_index in range(len(Y)-1):
    #         plt.plot([X[x_index],X[x_index+1],X[x_index+1],X[x_index]],[Y[y_index],Y[y_index],Y[y_index+1],Y[y_index+1]],color="blue")
    #         break
    #     break
    #
    # plt.show()

    ref1 = numpy.asarray([0.76,0.7,0.53])

    print (X[-1]-X[0])*(Y[-1]-Y[0])

    for x in range(X[0],X[-1]):
        # print x
        # x_temp = image[y]
        for y in range(Y[0],Y[-1]):
            dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(image[y][x],ref1)]))
            # temp = 0
            # i_temp = x_temp[y]
            # i_temp = image[y][x]
            # for i in range(3):
            #     temp += (i_temp[i]-ref1[i])**2
            # print i_temp
            # print math.sqrt(temp)
            # dist1 = math.sqrt(temp)
            # dist1 = math.sqrt(sum([(image[x][y][i]-ref1[i])**2 for i in [0,1,2]]))
            if dist1 > 0.20:
                # plt.plot(x,-y,"o",color="blue")

                if (min([abs(y-y_l) for y_l in Y]) > 1) and (min([abs(x-x_l) for x_l in X]) > 1):
                    r.append((x,-y))

    # plt.show()
    return numpy.asarray(r)

t = s()
# t.sort(key = lambda x:x[1])
# t.sort(key = lambda x:x[0])
print len(t)
x,y = zip(*t)

plt.plot(x,y,'.',color="blue")

plt.show()

from sklearn.cluster import DBSCAN
import scaling
db = DBSCAN(eps=2, min_samples=3).fit(t)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
unique_labels = set(labels)
colors = plt.cm.Spectral(numpy.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = t[class_member_mask]
    scaling.to_scale(xy)
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
    #          markeredgecolor='k', markersize=3)
    #
    #
    # print max(xy[:, 1])- min(xy[:, 1])



# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

# for x_index in range(len(X)-1):
#     for y_index in range(len(Y)-1):
#         # ul_x is the upper left corner - x coordinate
#         ul_x = X[x_index]
#         lr_x = X[x_index+1]
#
#         ul_y = Y[y_index]
#         lr_y = Y[y_index+1]
#
#         for (x,y) in t:
#             if x < ul_x:
#                 continue
#             elif x > lr_x:
#                 break
#             else:
#                 print (x >= ul_y) and (y <= lr_y)
#     break
#     #         plt.plot([X[x_index],X[x_index+1],X[x_index+1],X[x_index]],[Y[y_index],Y[y_index],Y[y_index+1],Y[y_index+1]],color="blue")