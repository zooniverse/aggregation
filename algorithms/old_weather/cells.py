__author__ = 'greg'
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy
import cPickle
import os
from neural_network import Network,load_data_wrapper
import cv2

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

net = cPickle.load(open(base_directory+"/Dropbox/neural.net","rb"))


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

image_file = cbook.get_sample_data("/home/ggdhines/Dropbox/vol072_237_0.png")
image = plt.imread(image_file)

# @jit
def s():
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)

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
                    r.append((x,y))

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


def p(a):
    for x in range(28):
        for y in range(28):

            if a[y*28+x] > 0:
                plt.plot(x,y,"o",color="blue")

    plt.xlim((-0.01,28))
    plt.ylim((28,-0.01))
    plt.show()

training_data, validation_data, test_data = load_data_wrapper()

# p(test_data[0][0])

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = t[class_member_mask]

    x_l,y_l = zip(*xy)
    for x,y in xy:
        plt.plot(x,-y,"o",color="blue")
    plt.show()

    sub_image = []
    for y in range(min(y_l),max(y_l)+1):
        temp = []
        for x in range(min(x_l),max(x_l)+1):
            temp.append(image[y][x])
        sub_image.append(temp)

    sub_image = numpy.array(sub_image)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    im = axes.imshow(sub_image)
    plt.show()


    x_range = max(x_l)-min(x_l)
    y_range = max(y_l)-min(y_l)

    if max(x_range,y_range) < 28:
        # always enlarge along the biggest axis
        if x_range > y_range:
            new_x = 28
            scale = 28/float(x_range)
            new_y = int(round(y_range*scale))
        else:
            new_y = 28
            scale = 28/float(y_range)
            new_x = int(round(x_range*scale))

        print new_y,new_x
        resized_image = cv2.resize(sub_image, (new_y,new_x), interpolation=cv2.INTER_CUBIC )
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        im = axes.imshow(resized_image)
        plt.show()

    elif max(x_range,y_range) > 28:
        # shrinking
        pass
    else:
        # just need to center
        pass

    continue

    assert False


    print min(x),min(y)
    print max(x),max(y)

    r = scaling.to_scale(xy)

    if r is not None:
        # # print type(r)
        # # print r.shape
        # print "here"
        # # print len(r)
        # # print len(net.feedforward(r)[0])
        l = net.feedforward(r)
        print numpy.argmax(l), numpy.max(l)/numpy.median(l)
        # print "->"
        p(r)