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
    pts = [tuple(p) for p in pts]
    retval = [0 for i in range(end_scale**2)]

    x_l,y_l = zip(*pts)

    min_y = min(y_l)
    min_x = min(x_l)

    y_l = [y-min(y_l) for y in y_l]
    x_l = [x-min(x_l) for x in x_l]

    # for (x,y) in zip(x_l,y_l):
    #     plt.plot(x,y,"o",color="blue")
    # plt.show()

    if min(y_l) == max(y_l):
        return
    if min(x_l) == max(x_l):
        return

    if (max(y_l)-min(y_l)) < (max(x_l) - min(x_l)):
        return
    else:
        height = max(y_l)
        width = max(x_l)
        scale = 28/float(height)
        print scale
        x_offset = (28/2)*(1-width/float(height))

        # what is the minimum x_1 values that maps to a x value in x_l
        # the smallest x value is 0 in x_l, so we have x_1 = scale*0+x_offset
        # want the smallest integer largest than this value
        x1_min = int(math.ceil(x_offset))
        # what is the largest x_1 value that maps to a value in x_l?
        # now we have x_1 = scale*width+x_x_offset
        x1_max = int(math.floor(scale*width+x_offset))

        # print pts

        for y1 in range(28):
            for x1 in range(x1_min,min(x1_max+1,28)):
                y0 = y1/scale
                x0 = (x1-x_offset)/scale

                l_x0 = int(math.floor(x0)) + min_x
                u_x0 = int(math.ceil(x0)) + min_x

                l_y0 = int(math.floor(y0)) + min_y
                u_y0 = int(math.ceil(y0)) + min_y

                # kernel based smoothing
                distances = []

                distances.append((x0-l_x0+min_x)**2+(y0-l_y0+min_y)**2)
                distances.append((x0-l_x0+min_x)**2+(y0-u_y0+min_y)**2)
                distances.append((x0-u_x0+min_x)**2+(y0-l_y0+min_y)**2)
                distances.append((x0-u_x0+min_x)**2+(y0-u_y0+min_y)**2)

                # print x0,y0
                # print l_x0,l_y0
                # print distances

                b = 0.5

                kernels = [math.exp(-d/float(2*b**2)) for d in distances]
                # print distances
                # print kernels
                v = []
                # kernels.append(math.exp())

                pos_neighbours = 0
                # print (l_x0,l_y0)
                # print pts
                # print (l_x0,l_y0)
                if (l_x0,l_y0) in pts:
                    pos_neighbours += 1
                    v.append(1)
                else:
                    v.append(0)
                if (l_x0,u_y0) in pts:
                    pos_neighbours += 1
                    v.append(1)
                else:
                    v.append(0)
                if (u_x0,l_y0) in pts:
                    pos_neighbours += 1
                    v.append(1)
                else:
                    v.append(0)
                if (u_x0,u_y0) in pts:
                    pos_neighbours += 1
                    v.append(1)
                else:
                    v.append(0)

                # print pos_neighbours/4.
                # print numpy.average(v,weights=kernels)
                # print "--"

                if numpy.average(v,weights=kernels) >= 0.5:
                    # plt.plot(x1,y1,"o",color="blue")
                    retval[(27-y1)*28+x1]=1

        # plt.xlim((-0.01,28))
        # plt.ylim((-0.01,28))
        # plt.show()

    return numpy.asarray(retval).reshape((784,1))

    # # make sure that the digit is vertical
    # assert starting_scale >= (max(x) - min(x))
    # print starting_scale
    # m = (end_scale)/float(starting_scale)
    # # print m
    # # assert False
    #
    # # jpegs are centered at 0,0 in the top left corner
    # y_offset = min(y)
    # x_offset = min(x)
    #
    # # we are blowing up the image so that the y-axis fits exactly, but we still want to center
    # # based on the x axis so that the center of the image maps to the end scale/2
    # # to not distort the number, we'll multiple both axes by the same scale and then add an additional x-offset
    # mid_x = (max(x)+min(x))/2.
    # b = (end_scale/2.-1)-m*mid_x
    # print b
    # # assert False
    # # x_offset += end_scale/2.-(min(x)+max(x))/2.*m
    # # print corner_x
    # # print x_offset
    # # for x_ in x:
    # #
    # #     print m*x_+x_offset
    #
    #
    # # m = (scale1-1)/float(scale2-1)
    #
    # # go through each of the points in the end scale
    # # and see where they map to in the original scale
    #
    # for x in range(end_scale):
    #     for y in range(end_scale):
    #         # where would this point be in the original scale?
    #         # m takes us from the original scale to the new one, so divide by m to go back
    #         # then add in the offset
    #         print y,y/m+y_offset
    #         print x,(x-b)/m
    #         print
    #         assert False
    #         continue
    #         l_x = math.floor(x*m)
    #         u_x = math.ceil(x*m)
    #
    #         l_y = math.floor(y*m)
    #         u_y = math.ceil(y*m)
    #
    #         ml_x = l_x/m
    #         ml_y = l_y/m
    #         mu_x = u_x/m
    #         mu_y = u_y/m
    #
    #
    #         v = []
    #
    #         distances = []
    #
    #         distances.append((x-ml_x)**2+(y-ml_y)**2)
    #         distances.append((x-ml_x)**2+(y-mu_y)**2)
    #         distances.append((x-mu_x)**2+(y-ml_y)**2)
    #         distances.append((x-mu_x)**2+(y-mu_y)**2)
    #
    #         b = 2
    #
    #         kernels = [math.exp(-d/float(2*b**2)) for d in distances]
    #         # kernels.append(math.exp())
    #
    #         v.append(training_data[0][index][l_x*28+l_y])
    #         v.append(training_data[0][index][l_x*28+u_y])
    #         v.append(training_data[0][index][u_x*28+u_y])
    #         v.append(training_data[0][index][u_x*28+l_y])
    #
    #         # print numpy.average(v,weights=kernels)
    #         # print numpy.mean(v)
    #         # print
    #         if numpy.average(v,weights=kernels) > 0.1:
    #             print v
    #             print kernels
    #             # plt.plot(y,-x,"o",color="blue")
    #
    # # plt.show()