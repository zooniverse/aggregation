__author__ = 'greg'
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy
from scipy import spatial
from PIL import Image
import pytesseract






start_x = 585.5
end_x = 3265.5
horizontal_lines = [(1271.5,1288.5),(1368.5,1384),(1424,1440),(1481.5,1497),(1537.5,1552),(1593,1607.5),(1649,1663),(1705.5,1718.5),(1762.5,1775.5),(1819,1830.5),(1875,1886),(1931,1942),(1987,1998)]

vertical_lines = [586,778.5,882,948,1049.5,1117,1293,1456.5,1557.5,1758.5,1852,1959,2058.5,2157.5,2325,2559.5,2731,2855.5,2959,3095,3266]
start_y = 1271.5
end_y = 1998

# def intersection(((y1,y2),(x1,x2)),x_):
#     m = (y2-y1)/float(x2-x1)
#     b = y2 - m*x2
#
#     y_intersect = m*x_+b
#
#     return x_,y_intersect

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

# fig = plt.figure()
# axes = fig.add_subplot(1, 1, 1)
# im = axes.imshow(image)

# for y_l in horizontal_lines:
#     for x in vertical_lines:
#         x_intersect,y_intersect = intersection((y_l,(start_x,end_x)),x)
#         plt.plot(x_intersect,y_intersect,"o",color="red")
#
# plt.show()

ref1 = [218,219,213]

globalX_l = []
globalY_l = []

for y in range(1271,2000):
    # print x
    # x_temp = image[y]
    for x in range(585,3265):
        dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(image[y][x],ref1)]))

        if dist1 > 25:
            globalX_l.append(x)
            globalY_l.append(y)
# plt.plot(X_l,Y_l,"o",color="blue")
#
# plt.ylim((2000,1271))
# plt.xlim((585,3265))


global_tree = spatial.KDTree(zip(globalX_l,globalY_l))

# corner = 2858,1934

def corner_correction(corner):
    X_l2 = []
    Y_l2 = []

    pt_indices = global_tree.query_ball_point(corner,r=30)
    for i in pt_indices:
        x = globalX_l[i]
        y = globalY_l[i]
        X_l2.append(x)
        Y_l2.append(y)

        # plt.plot(x,y,"o",color="red")

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


def deter(a,b,c,d):
    return a*d - c*b


def line_highlight(c1,c2,x_offset,y_offset,local_pts,line_type):
    assert isinstance(local_pts,list)
    x1,y1 = c1
    x2,y2 = c2

    x1 -= x_offset
    x2 -= x_offset
    y1 -= y_offset
    y2 -= y_offset



    # print "here"
    # horiz. lines
    if line_type in ["top","bottom"]:
        m = (y2-y1)/float(x2-x1)
        b = y1 - m*x1
        if line_type == "top":
            local_pts.sort(key = lambda pt:pt[1])

            for pt_index,(x,y) in enumerate(local_pts):
                y_prime = m*x+b

                if math.fabs(y-y_prime) > 4:
                    return local_pts[pt_index:]
        elif line_type == "bottom":
            local_pts.sort(key = lambda pt:pt[1],reverse=True)

            for pt_index,(x,y) in enumerate(local_pts):
                y_prime = m*x+b

                if math.fabs(y-y_prime) > 4:
                    return local_pts[pt_index:]
    else:
        # vertical lines
        if line_type == "left":
            local_pts.sort(key = lambda pt:pt[0])

            for pt_index,(x,y) in enumerate(local_pts):
                if math.fabs(x-x1) > 4:
                    return local_pts[pt_index:]
        elif line_type == "right":
            local_pts.sort(key = lambda pt:pt[0],reverse=True)

            for pt_index,(x,y) in enumerate(local_pts):

                if math.fabs(x-x1) > 4:
                    return local_pts[pt_index:]




    assert False


def intersection(h_l,v_l):
    x1 = start_x
    y1 = h_l[0]

    x2 = end_x
    y2 = h_l[1]

    x3 = v_l
    y3 = start_y

    x4 = v_l
    y4 = end_y

    d1 = deter(x1,y1,x2,y2)
    d2 = deter(x1,1,x2,1)
    d3 = deter(x3,y3,x4,y4)
    d4 = deter(x3,1,x4,1)

    D1 = deter(d1,d2,d3,d4)

    d5 = deter(y1,1,y2,1)
    d6 = deter(y3,1,y4,1)

    D2 = deter(d1,d5,d3,d6)

    d7 = deter(x3,1,x4,1)
    d8 = deter(y3,1,y4,1)

    D3 = deter(d2,d5,d7,d8)

    intersect_x = int(D1/D3)
    intersect_y = int(D2/D3)

    return intersect_x,intersect_y

# plt.show()

end_size = 28


def shrink(pts):
    tree = spatial.KDTree(pts)
    X_l,Y_l = zip(*pts)
    max_x,min_x = max(X_l),min(X_l)
    max_y,min_y = max(Y_l),min(Y_l)

    ret_pts = []

    for x in range(end_size):
        for y in range(end_size):
            mapped_x = x/float(end_size)*(max_x-min_x) + min_x
            mapped_y = y/float(end_size)*(max_y-min_y) + min_y

            neighbours_index = tree.query_ball_point((mapped_x,mapped_y),r=3)

            distances = []

            for ii in neighbours_index:
                x_t, y_t = pts[ii]
                # plt.plot(x_t,y_t,"o",color="green")
                distances.append(math.sqrt((mapped_x-x_t)**2+(mapped_y-y_t)**2))
            # plt.plot(mapped_x,mapped_y,"o",color="red")
            # print distances
            k = [math.exp(-d/float(2*b**0.05)) for d in distances]
            if (k != []) and (numpy.mean(k) > 0.45):
                ret_pts.append((x,y))

    return ret_pts

test_cases = []
test_digits = []

from sklearn.cluster import DBSCAN
done = False

for h_index,h_l in enumerate(horizontal_lines[:-1]):
    if done:
        break
    for v_index,v_l in enumerate(vertical_lines[:-1]):
        ul_x,ul_y = intersection(h_l,v_l)
        ul_x,ul_y = corner_correction((ul_x,ul_y))

        ll_x,ll_y = intersection(horizontal_lines[h_index+1],v_l)
        ll_x,ll_y = corner_correction((ll_x,ll_y))

        ur_x,ur_y = intersection(h_l,vertical_lines[v_index+1])
        ur_x,ur_y = corner_correction((ur_x,ur_y))

        lr_x,lr_y = intersection(horizontal_lines[h_index+1],vertical_lines[v_index+1])
        lr_x,lr_y = corner_correction((lr_x,lr_y))

        x_0 = min(ul_x,ll_x)
        # print h_l,v_l
        # print horizontal_lines[h_index+1],vertical_lines[v_index+1]
        # print "==----"
        # print (ul_x,ll_x)
        x_1 = max(ur_x,lr_x)
        # print (ur_x,lr_x)

        y_0 = min(ul_y,ur_y)
        # print (ul_y,ur_y)
        y_1 = max(ll_y,lr_y)
        # print (ll_y,lr_y)
        # print

        rows = numpy.array([[y_0+i for j in range(x_1-x_0+1)] for i in range(y_1-y_0+1)])
        columns = numpy.array([[x_0+j for j in range(x_1-x_0+1)] for i in range(y_1-y_0+1)])
        subimage = image[rows,columns]

        # for i in range(subimage.shape[0]):
        #     for j in range(subimage.shape[1]):
        #         dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(subimage[i][j],ref1)]))
        #         if dist1 < 25:
        #             subimage[i][j] = [255,255,255]

        r = Image.fromarray(subimage)
        # r.save("/home/greg/temp.jpg")
        # r = Image.open("/home/greg/temp.jpg")
        # r.save("/home/greg/temp.pdf","PDF", Quality = 100)
        # print(pytesseract.image_to_string(Image.open("/home/greg/output.pdf")))
        #
        # fig = plt.figure()
        # axes = fig.add_subplot(1, 1, 1)
        # im = axes.imshow(subimage)
        # plt.show()

        local_pts = []
        for y in range(len(subimage)):
            for x in range(len(subimage[0])):
                dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(subimage[y][x],ref1)]))
                if dist1 > 25:
                    # plt.plot(x,-y,"o",color="blue")
                    local_pts.append((x,y))
        # plt.show()
        extra = 2
        # plt.ylim((len(subimage)+extra,0-extra))
        # plt.xlim((0-extra,len(subimage[0])+extra))
        # plt.plot([ul_x-x_0,ur_x-x_0],[ul_y-y_0,ur_y-y_0],"-",color="red")
        # plt.plot([ul_x-x_0,ll_x-x_0],[ul_y-y_0,ll_y-y_0],"-",color="red")
        # plt.plot([ll_x-x_0,lr_x-x_0],[ll_y-y_0,lr_y-y_0],"-",color="red")
        # plt.plot([lr_x-x_0,ur_x-x_0],[lr_y-y_0,ur_y-y_0],"-",color="red")


        # plt.plot(ul_x-x_0,ul_y-y_0,"o",color="yellow")

        ur_x,ur_y = corner_correction((ur_x,ur_y))
        # plt.plot(ur_x-x_0,ur_y-y_0,"o",color="yellow")

        local_pts = line_highlight((ul_x,ul_y),(ur_x,ur_y),x_0,y_0,local_pts,"top")
        local_pts = line_highlight((ll_x,ll_y),(lr_x,lr_y),x_0,y_0,local_pts,"bottom")
        local_pts = line_highlight((ll_x,ll_y),(ll_x,ll_y),x_0,y_0,local_pts,"left")
        local_pts = line_highlight((ur_x,ur_y),(lr_x,lr_y),x_0,y_0,local_pts,"right")

        # x,y = corner_correction((ll_x,ll_y))
        # plt.plot(x-x_0,y-y_0,"o",color="yellow")
        # x,y = corner_correction((ur_x,ur_y))
        # plt.plot(x-x_0,y-y_0,"o",color="yellow")
        # x,y = corner_correction((lr_x,lr_y))
        # plt.plot(x-x_0,y-y_0,"o",color="yellow")



        # plt.show()

        x_pts,y_pts = zip(*local_pts)
        # plt.plot(x_pts,y_pts,"o",color="green")
        # plt.ylim((len(subimage)+extra,0-extra))
        # plt.xlim((0-extra,len(subimage[0])+extra))
        # plt.show()

        local_pts = numpy.asarray(local_pts)


        # plt.plot(intersect_x,intersect_y,"o",color="yellow")

        db = DBSCAN(eps=3, min_samples=20).fit(local_pts)
        labels = db.labels_
        unique_labels = set(labels)
        colors = plt.cm.Spectral(numpy.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
                continue
            else:
                col = "blue"
            class_member_mask = (labels == k)

            xy = local_pts[class_member_mask]
            X_l,Y_l = zip(*xy)
            max_x = max(X_l)
            max_y = max(Y_l)

            min_x = min(X_l)
            min_y = min(Y_l)

            desired_height = 20.

            width_ratio = (max_x-min_x)/desired_height
            height_ratio = (max_y-min_y)/desired_height

            if width_ratio > height_ratio:
                # wider than taller
                # todo - probably not a digit
                width = int(desired_height)
                height = int(desired_height*(max_y-min_y)/float(max_x-min_x))
            else:
                height = int(desired_height)
                # print (max_y-max_y)/float(max_x-min_x)
                width = int(desired_height*(max_x-min_x)/float(max_y-min_y))

            template = [[[1,1,1] for i in range(min_x,max_x+1)] for j in range(min_y,max_y+1)]
            for x,y in xy:
                template[y-min_y][x-min_x] = subimage[y][x]
                # plt.plot(x,-y,"o",color="blue")

            # print type(xy)
            # xy = list(xy)
            # for x in range(min_x,max_x+1):
            #     for y in range(min_y,max_y+1):
            #         if (x,y) in xy:
            #             # template[y-min_y][x-min_x] = subimage[y][x]

                    # else:
                    #     template[y-min_y][x-min_x] = [1,1,1]

            # template = numpy.asarray(template)
            # print template
            # print type(template)

            # print type(template)
            # print width,height
            # print (min_x,max_x),(min_y,max_y)
            # for a in template:
            #     for b in a:
            #         print b

            digit_image = Image.fromarray(numpy.uint8(numpy.asarray(template)))
            # plt.show()

            digit_image = digit_image.resize((width,height),Image.ANTIALIAS)
            digit_image = digit_image.convert('L')

            # # we need to center subject
            # if height == 28:
            #     # center width wise
            #
            #     y_offset = 0
            # else:
            #
            #     x_offset = 0

            x_offset = int(28/2 - width/2)
            y_offset = int(28/2 - height/2)

            ref2 = [1,1,1]
            digit_array = numpy.asarray(digit_image)
            print "****"
            centered_array = [0 for i in range(28**2)]

            darkest_pixel = 0
            for y in range(len(digit_array)):
                for x in range(len(digit_array[0])):
                    darkest_pixel = max(darkest_pixel,digit_array[y][x])

            darkest_pixel = max(darkest_pixel,100)

            for y in range(len(digit_array)):
                for x in range(len(digit_array[0])):
                    # dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(digit_array[y][x],ref1)]))
                    # if dist1 > 10:
                    # if digit_array[y][x] > 0.4:
                    #     plt.plot(x+x_offset,y+y_offset,"o",color="blue")
                    # digit_array[y][x] = digit_array[y][x]/255.
                    if digit_array[y][x] > 10:
                        centered_array[(y+y_offset)*28+(x+x_offset)] = digit_array[y][x]/float(darkest_pixel)
                    else:
                        centered_array[(y+y_offset)*28+(x+x_offset)] = 0
                    # assert (digit_array[y][x]/255.) <= 1

            print digit_array
            digit = raw_input("enter digit: ")
            if digit == "":
                done = True
                break
            elif digit != "s":
                centered_array = numpy.asarray(centered_array).reshape((28**2,1))
                test_cases.append(centered_array)
                test_digits.append(int(digit))

            for index,i in enumerate(centered_array):
                if i > 0:
                    x = index%28
                    y = index/28
                    plt.plot(x,y,"o",color="blue")

            plt.ylim((28,0))
            plt.xlim((0,28))
            plt.show()


            # plt.ylim((28,0))
            # plt.xlim((0,28))

            # fig = plt.figure()
            # axes = fig.add_subplot(1, 1, 1)
            # im = axes.imshow(digit_image)
            # plt.show()
            # digit_image.save("/home/greg/test.jpg")
            # plt.imshow(digit_image)
            # plt.show()
        if done:
            break

import neural_network
training_data, validation_data, test_data = neural_network.load_data_wrapper()
net = neural_network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=zip(test_cases,test_digits))