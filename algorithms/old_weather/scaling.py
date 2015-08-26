__author__ = 'ggdhines'
import gzip
import cPickle
import matplotlib.pyplot as plt
import math
import numpy

# f = gzip.open('/home/greg/github/neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
# training_data, validation_data, test_data = cPickle.load(f)
# f.close()
#
# index = 0
#
# # scale1 = 28
# # scale2 = 40
#
# for x in range(28):
#     for y in range(28):
#         if training_data[0][index][x*28+y] > 0:
#             plt.plot(y,-x,"o",color="blue")
#
# print training_data[1][index]
# plt.show()
#
# for x in range(28):
#     for y in range(28):
#         if training_data[0][index][y*28+x] > 0:
#             plt.plot(x,-y,"o",color="blue")
#
# print training_data[1][index]
# plt.show()
#
end_scale = 28
def to_scale(pts):
    retval = [0 for i in range(end_scale**2)]

    x,y = zip(*pts)
    starting_scale = max(y)-min(y)
    if (max(y)-min(y)) < (max(x) - min(x)):
        return
    # make sure that the digit is vertical
    assert starting_scale >= (max(x) - min(x))
    print starting_scale
    m = (end_scale)/float(starting_scale)
    # print m
    # assert False

    # jpegs are centered at 0,0 in the top left corner
    y_offset = min(y)
    x_offset = min(x)

    # we are blowing up the image so that the y-axis fits exactly, but we still want to center
    # based on the x axis so that the center of the image maps to the end scale/2
    # to not distort the number, we'll multiple both axes by the same scale and then add an additional x-offset
    mid_x = (max(x)+min(x))/2.
    b = (end_scale/2.-1)-m*mid_x
    print b
    # assert False
    # x_offset += end_scale/2.-(min(x)+max(x))/2.*m
    # print corner_x
    # print x_offset
    # for x_ in x:
    #
    #     print m*x_+x_offset


    # m = (scale1-1)/float(scale2-1)

    # go through each of the points in the end scale
    # and see where they map to in the original scale

    for x in range(end_scale):
        for y in range(end_scale):
            # where would this point be in the original scale?
            # m takes us from the original scale to the new one, so divide by m to go back
            # then add in the offset
            print y,y/m+y_offset
            print x,(x-b)/m
            print
            assert False
            continue
            l_x = math.floor(x*m)
            u_x = math.ceil(x*m)

            l_y = math.floor(y*m)
            u_y = math.ceil(y*m)

            ml_x = l_x/m
            ml_y = l_y/m
            mu_x = u_x/m
            mu_y = u_y/m


            v = []

            distances = []

            distances.append((x-ml_x)**2+(y-ml_y)**2)
            distances.append((x-ml_x)**2+(y-mu_y)**2)
            distances.append((x-mu_x)**2+(y-ml_y)**2)
            distances.append((x-mu_x)**2+(y-mu_y)**2)

            b = 2

            kernels = [math.exp(-d/float(2*b**2)) for d in distances]
            # kernels.append(math.exp())

            v.append(training_data[0][index][l_x*28+l_y])
            v.append(training_data[0][index][l_x*28+u_y])
            v.append(training_data[0][index][u_x*28+u_y])
            v.append(training_data[0][index][u_x*28+l_y])

            # print numpy.average(v,weights=kernels)
            # print numpy.mean(v)
            # print
            if numpy.average(v,weights=kernels) > 0.1:
                print v
                print kernels
                # plt.plot(y,-x,"o",color="blue")

    # plt.show()