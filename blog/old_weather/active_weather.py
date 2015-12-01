# import matplotlib
# matplotlib.use('WXAgg')
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data
from skimage.data import load
from skimage.color import rgb2gray
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import math
import matplotlib.path as mplPath

def deter(a,b,c,d):
    return a*d - c*b

def does_intersect(l1,l2):
    (x1,y1),(x2,y2) = l1
    (x3,y3),(x4,y4) = l2

    l1_lower_x = min(x1,x2)
    l1_upper_x = max(x1,x2)
    l2_lower_x = min(x3,x4)
    l2_upper_x = max(x3,x4)

    l1_lower_y = min(y1,y2)
    l1_upper_y = max(y1,y2)
    l2_lower_y = min(y3,y4)
    l2_upper_y = max(y3,y4)

    x_inter,y_inter = intersection(l1,l2)
    valid_x = (l1_lower_x <= x_inter) and (l2_lower_x <= x_inter) and (x_inter <= l1_upper_x) and (x_inter <= l2_upper_x)
    valid_y = (l1_lower_y <= y_inter) and (l2_lower_y <= y_inter) and (y_inter <= l1_upper_y) and (y_inter <= l2_upper_y)

    return valid_x and valid_y

def intersection(h_line,v_line):
    (x1,y1),(x2,y2) = h_line
    (x3,y3),(x4,y4) = v_line

    # do we have a completely vertical line
    if x3 == x4:
        intersect_x = x3

        m = (y2-y1)/float(x2-x1)
        b = y2 - m*x2
        intersect_y = m*intersect_x+b
    else:
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

        intersect_x = D1/float(D3)
        intersect_y = D2/float(D3)

    return intersect_x,intersect_y

def multi_intersection(h_multi_line,v_multi_line):
    for h_index in range(len(h_multi_line)-1):
        h_line = h_multi_line[h_index:h_index+2]

        for v_index in range(len(v_multi_line)-1):
            v_line = v_multi_line[v_index:v_index+2]

            if does_intersect(h_line,v_line):
                return h_index,v_index,intersection(h_line,v_line)

    assert False

def hesse_line(line_seg):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """


    (x1,y1),(x2,y2) = line_seg

    # x2 += random.uniform(-0.0001,0.0001)
    # x1 += random.uniform(-0.0001,0.0001)

    dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

    try:
        tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
        theta = math.atan(tan_theta)
    except ZeroDivisionError:
        theta = math.pi/2.

    return dist,theta


# image = data.camera()
image = load("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG")
# image = rgb2gray(image)
# aws s3 ls s3://zooniverse-static/old-weather-2015/Distant_Seas/Navy/Bear_AG-29_/Bear-AG-29-1939/
# img = cv2.imread("/home/ggdhines/Databases/old_weather/test_cases/Bear-AG-29-1939-0191.JPG",0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,25,150,apertureSize = 3)
# cv2.imwrite('/home/ggdhines/1.jpg',edges)

lines = probabilistic_hough_line(edges, threshold=8, line_length=8,line_gap=1)
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(52,78)
ax1.imshow(edges)

plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
assert False

horiz_list = []
horiz_intercepts = []

vert_list = []
vert_intercepts = []



big_lower_x = 559
big_upper_x = 3245
big_lower_y = 1292
big_upper_y = 2014

horiz_lines = []
vert_lines = []

for line in lines:
    p0, p1 = line
    X = p0[0],p1[0]
    Y = p0[1],p1[1]

    if (min(X) >= 559) and (max(X) <= 3245) and (min(Y) >= 1292) and (max(Y) <= 2014):
        d,t = hesse_line(line)
        if math.fabs(t) <= 0.1:
            horiz_list.append(line)
            # hesse_list.append(hesse_line(line))

            m = (Y[0]-Y[1])/float(X[0]-X[1])
            b = Y[0]-m*X[0]
            horiz_intercepts.append(b+m*big_lower_x)
            horiz_lines.append(line)
        elif math.fabs(t-math.pi/2.) <= 0.1:
            vert_list.append(line)
            m = (X[0]-X[1])/float(Y[0]-Y[1])
            b = X[0]-m*Y[0]
            vert_intercepts.append(b+m*big_lower_y)
            vert_lines.append(line)


        # ax1.plot(X, Y,color="red")
    # else:
    #     print min(X)
    # print hesse_line(line)

clusters = [-1 for i in range(len(horiz_lines))]
clusters_count = -1

horiz_lines.sort(key = lambda x:x[0][0])
for l_index in range(len(horiz_lines)-1):
    for l2_index in range(l_index+1,len(horiz_lines)):
        line_1 = horiz_lines[l_index]
        line_2 = horiz_lines[l2_index]

        (x1,y1),(x2,y2) = line_1
        (x3,y3),(x4,y4) = line_2

        assert x1 <= x3

        if x3 <= x2:
            dist = y4 - y2
            x_dist = 0
        else:
            m = (y2-y1)/float(x2-x1)
            b = y2 - m*x2

            d1 = (m*x3+b)-y3
            d2 = (m*x4+b)-y4
            dist = max(d1,d2)

            x_dist = math.sqrt((x3-x2)**2+(y3-y2)**2)

        # print dist
        if (math.fabs(dist) < 0.3) and (x_dist < 100):
            if (clusters[l_index] == -1) and (clusters[l2_index] == -1):
                clusters_count += 1
                clusters[l_index] = clusters_count
                clusters[l2_index] = clusters_count
            elif clusters[l_index] == -1:
                clusters[l_index] = clusters[l2_index]
            elif clusters[l2_index] == -1:
                clusters[l2_index] = clusters[l_index]
            else:
                new_cluster_id = max(clusters[l2_index], clusters[l_index])
                old_cluster_id = min(clusters[l2_index], clusters[l_index])
                for j,c in enumerate(clusters):
                    if c == old_cluster_id:
                        clusters[j] = new_cluster_id


for c in range(clusters_count):
    pts = []
    dist = 0
    for i,l in enumerate(horiz_lines):
        if clusters[i] == c:
            pts.extend(l)
            (x1,y1),(x2,y2) = l
            dist += x2-x1
    # print dist/float(big_upper_x-big_lower_x)
    # if dist/float(big_upper_x-big_lower_x) < 0.6:
    #     continue
    if pts == []:
        continue
    pts.sort(key = lambda x:x[0])
    X,Y = zip(*pts)
    plt.plot(X,Y)

plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
assert False

from sklearn.cluster import DBSCAN

# def fil_gaps(line_segments,horiz=True):
#     first_pt = line_segments[0][0]
#     last_pt = line_segments[-1][1]
#
#     for l_index in range(len(line_segments)-1):
#         l1 = line_segments[l_index]
#         l2 = line_segments[l_index+1]
#
#         line_segments.append((l1[1],l2[0]))
#
#     return line_segments

# functions for sorting lines - has to deal with the fact that either lb_lines or ub_lines could be empty
def lowest_y((lb_lines,ub_lines)):
    if lb_lines == []:
        assert ub_lines != []
        return ub_lines[0][1]
    else:
        return lb_lines[0][1]


def lowest_x((lb_lines,ub_lines)):
    if lb_lines == []:
        assert ub_lines != []
        return ub_lines[0][0]
    else:
        return lb_lines[0][0]

# from
# http://geospatialpython.com/2011/01/point-in-polygon.html
# def point_in_poly(x,y,poly):
#
#     n = len(poly)
#     inside = False
#
#     p1x,p1y = poly[0]
#     for i in range(n+1):
#         p2x,p2y = poly[i % n]
#         if y > min(p1y,p2y):
#             if y <= max(p1y,p2y):
#                 if x <= max(p1x,p2x):
#                     if p1y != p2y:
#                         xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
#                     if p1x == p2x or x <= xints:
#                         inside = not inside
#         p1x,p1y = p2x,p2y
#
#     return inside


def __pixel_generator__(boundary):
    most_common_colour = [222,222,220]
    X,Y = zip(*boundary)
    x_min = int(math.ceil(min(X)))
    x_max = int(math.floor(max(X)))

    y_min = int(math.ceil(min(Y)))
    y_max = int(math.floor(max(Y)))


    bbPath = mplPath.Path(np.asarray(boundary))

    for x in range(x_min,x_max+1):
        for y in range(y_min,y_max+1):

            if bbPath.contains_point((x,y)):
                dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(image[y][x],most_common_colour)]))
                if dist > 40:
                    plt.plot(x,y,"o",color="blue")

    plt.xlim((x_min,x_max))
    plt.ylim((y_max,y_min))


def analysis(lines,intercepts,horiz=True):
    retval = []
    # d_list,t_list = zip(*hesse_list)
    # min_dist = min(intercepts)
    # max_dist = max(intercepts)
    # min_theta = min(t_list)
    # max_theta = max(t_list)

    # normalized_d = [[(d-min_dist)/float(max_dist-min_dist),] for d in intercepts]
    # normalized_t = [(t-min_theta)/float(max_theta-min_theta) for t in t_list]
    # print normalized_d
    # print normalized_t

    X = np.asarray([[i,] for i in intercepts])
    # print X
    db = DBSCAN(eps=1, min_samples=1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

            continue

        class_indices = [i for (i,l) in enumerate(labels) if l == k]
        class_size = sum([1 for (i,l) in enumerate(labels) if l == k])
        if class_size <= 2:
            continue
        # print class_indices
        print len(class_indices)
        print [intercepts[i] for (i,l) in enumerate(labels) if l == k]

        multiline = []

        for i in class_indices:

            p0, p1 = lines[i]

            # if vertical - flip, then we'll flip back later
            if not horiz:
                p0 = [p0[1],p0[0]]
                p1 = [p1[1],p1[0]]

            # insert in increasing "X" order (quotation marks refer to the above flipping)
            # if horiz, sort by increasing X values

            if p0[0] < p1[0]:
                multiline.append([list(p0),list(p1)])
            else:

                multiline.append([list(p1),list(p0)])

        # sort so that we can add lines in order
        multiline.sort(key = lambda pt:pt[0][0])

        lb = set()
        ub = set()

        lb_lines = []
        ub_lines = []

        lower_distance = 0
        upper_distance = 0

        # note that these line segments will often not be overlapping but we want the ones that are since they will
        # they may provide upper and lower bounds on a line.
        for l_index,line in enumerate(multiline):
            for l2_index,line_2 in list(enumerate(multiline)):
                if l_index == l2_index:
                    continue

                #make sure that they are overlapping
                (x1,y1),(x2,y2) = line
                (x3,y3),(x4,y4) = line_2

                # d1 = (x2-x1)
                # d2 = (x4-x3)

                # if line starts before line_2, make sure that it ends after line_2 has started
                # or line starts before line_2 ends
                # these two cases cover overlaps
                if ((x1 < x3) and (x2 > x3)) or ((x1 >= x3) and (x1 <= x4)):
                    # we have an overlap
                    # the second case is slightly redundant but we have some strange cases
                    if (y1 < y3) and (y2 < y4):
                        lb.add(l_index)
                        ub.add(l2_index)

                        # lower_distance += d1
                        # upper_distance += d2
                    elif (y1 > y3) and (y2 > y4):
                        lb.add(l2_index)
                        ub.add(l_index)

                        # lower_distance += d2
                        # upper_distance += d1

        lb = sorted(list(lb))
        ub = sorted(list(ub))
        for i in lb:
            (x1,y1),(x2,y2) = multiline[i]
            if horiz:
                # lb_lines.append(multiline[i])
                lb_lines.extend(multiline[i])
                d = x2 - x1
            else:
                # else flip

                lb_lines.extend(((y1,x1),(y2,x2)))
                d = x2-x1
            lower_distance += d

        for i in ub:
            (x1,y1),(x2,y2) = multiline[i]
            if horiz:
                ub_lines.extend(multiline[i])
                d = x2 - x1
            else:
                # else flip

                ub_lines.extend([(y1,x1),(y2,x2)])
                d = x2 - x1
            upper_distance += d



        if horiz:
            D = big_upper_x-big_lower_x
        else:
            D = big_upper_y-big_lower_y

        if lb_lines != []:
            X,Y = zip(*lb_lines)
            x_min = min(X)
            x_max = max(X)
            p1 = lower_distance/float(D)
        else:
            p1 = -1



        if ub_lines != []:
            X,Y = zip(*ub_lines)
            x_min = min(X)
            x_max = max(X)
            p2 = upper_distance/float(D)
        else:
            p2 = -1

        if max(p1,p2) < 0.5:
            continue

        if horiz:
            if lb_lines != []:
                first_y = lb_lines[0][1]
                last_y = lb_lines[-1][1]

                lb_lines.insert(0,(big_lower_x,first_y))
                lb_lines.append((big_upper_x,last_y))

            if ub_lines != []:
                first_y = ub_lines[0][1]
                last_y = ub_lines[-1][1]

                ub_lines.insert(0,(big_lower_x,first_y))
                ub_lines.append((big_upper_x,last_y))
        else:
            if lb_lines != []:
                first_x = lb_lines[0][0]
                last_x = lb_lines[-1][0]

                lb_lines.insert(0,(first_x,big_lower_y))
                lb_lines.append((last_x,big_upper_y))

            if ub_lines != []:
                first_x = ub_lines[0][0]
                last_x = ub_lines[-1][0]

                ub_lines.insert(0,(first_x,big_lower_y))
                ub_lines.append((last_x,big_upper_y))

        if True:
            if lb_lines != []:
                X,Y = zip(*lb_lines)
                plt.plot(X,Y,"-",color="red")
            if ub_lines != []:
                X,Y = zip(*ub_lines)
                plt.plot(X,Y,"-",color="blue")

            for i,line in enumerate(multiline):
                if (i not in lb) and (i not in ub):
                    if horiz:
                        (x1,y1),(x2,y2) = line
                    else:
                        (y1,x1),(y2,x2) = line

                    plt.plot([x1,x2],[y1,y2],"-",color="green")
        if lb_lines != [] or ub_lines != []:
            # todo - what to do when they are both empty
            retval.append((lb_lines,ub_lines))

    if horiz:
        # assert isinstance(retval[0][0][0][1],int)
        retval.sort(key = lambda l:lowest_y(l))
    else:
        # assert isinstance(retval[0][0][0][0],int)
        retval.sort(key = lambda l:lowest_x(l))
    return retval

h_lines = analysis(horiz_list,horiz_intercepts,horiz=True)
# assert False
v_lines = analysis(vert_list,vert_intercepts,horiz=False)

# ax1.set_title('Probabilistic Hough')
ax1.set_axis_off()
ax1.set_adjustable('box-forced')



plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
plt.close()


for row_index in range(len(h_lines)-1):
    if [] in h_lines[row_index]:
        continue

    for column_index in range(len(v_lines)-1):
        if column_index == 0:
            continue
        if [] in v_lines[column_index]:
            continue
        # start with the top left
        lower_horiz = h_lines[row_index][1]
        lower_vert = v_lines[column_index][1]
        # print lower_horiz
        # print lower_vert
        x1_index,y1_index,(x1,y1) = multi_intersection(lower_horiz,lower_vert)
        plt.plot(x1,y1,"o",color="yellow")

        upper_vert = v_lines[column_index+1][0]
        x2_index,y2_index,(x2,y2) = multi_intersection(lower_horiz,upper_vert)

        upper_horiz = h_lines[row_index+1][0]
        x3_index,y3_index,(x3,y3) = multi_intersection(upper_horiz,upper_vert)

        x4_index,y4_index,(x4,y4) = multi_intersection(upper_horiz,lower_vert)

        # let's draw out the bounding box/polygon
        offset = 0.01
        cell_boundries = [(x1+offset,y1+offset)]
        cell_boundries.extend([(x,y+offset) for (x,y) in lower_horiz[x1_index+1:x2_index+1]])
        cell_boundries.append((x2-offset,y2+offset))
        cell_boundries.extend([(x-offset,y) for (x,y) in upper_vert[y2_index+1:y3_index+1]])
        cell_boundries.append((x3-offset,y3-offset))
        cell_boundries.extend([(x,y-offset) for (x,y) in reversed(upper_horiz[x3_index+1:x4_index+1])])
        cell_boundries.append((x4+offset,y4-offset))
        cell_boundries.extend([(x+offset,y) for (x,y) in reversed(lower_vert[y1_index+1:y4_index+1])])
        cell_boundries.append((x1+offset,y1+offset))


        plt.close()
        X,Y = zip(*cell_boundries)
        plt.plot(X,Y,color="red")
        __pixel_generator__(cell_boundries)
        plt.show()
        assert False

        # now convert from points back to the list



