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

def multi_intersection(h_line,v_line):
    for h in h_line:
        for v in v_line:
            if does_intersect(h,v):
                x,y = intersection(h,v)
                print h
                print v
                print
                plt.plot(x,y,"o",color ="green")

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

lines = probabilistic_hough_line(edges, threshold=5, line_length=50,line_gap=0)
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(52,78)
ax1.imshow(image)

horiz_list = []
horiz_intercepts = []

vert_list = []
vert_intercepts = []

big_lower_x = 559
big_upper_x = 3245
big_lower_y = 1292
big_upper_y = 2014

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
            horiz_intercepts.append(b)
        else:
            vert_list.append(line)
            m = (X[0]-X[1])/float(Y[0]-Y[1])
            b = X[0]-m*Y[0]
            vert_intercepts.append(b)


        # ax1.plot(X, Y,color="red")
    # else:
    #     print min(X)
    # print hesse_line(line)

from sklearn.cluster import DBSCAN

def fil_gaps(line_segments,horiz=True):
    first_pt = line_segments[0][0]
    last_pt = line_segments[-1][1]

    for l_index in range(len(line_segments)-1):
        l1 = line_segments[l_index]
        l2 = line_segments[l_index+1]

        line_segments.append((l1[1],l2[0]))

    return line_segments


def analysis(lines,intercepts,horiz=True):
    retval = []
    # d_list,t_list = zip(*hesse_list)
    min_dist = min(intercepts)
    max_dist = max(intercepts)
    # min_theta = min(t_list)
    # max_theta = max(t_list)

    normalized_d = [[(d-min_dist)/float(max_dist-min_dist),] for d in intercepts]
    # normalized_t = [(t-min_theta)/float(max_theta-min_theta) for t in t_list]
    # print normalized_d
    # print normalized_t

    X = np.asarray([[i,] for i in intercepts])
    # print X
    db = DBSCAN(eps=15, min_samples=1).fit(X)
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
        # print class_indices
        print len(class_indices)

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

        # note that these line segments will often not be overlapping but we want the ones that are since they will
        # they may provide upper and lower bounds on a line.
        for l_index,line in enumerate(multiline):
            for l2_index,line_2 in list(enumerate(multiline)):
                if l_index == l2_index:
                    continue

                #make sure that they are overlapping
                (x1,y1),(x2,y2) = line
                (x3,y3),(x4,y4) = line_2

                # if line starts before line_2, make sure that it ends after line_2 has started
                # or line starts before line_2 ends
                # these two cases cover overlaps
                if ((x1 < x3) and (x2 > x3)) or ((x1 >= x3) and (x1 < x4)):
                    # we have an overlap
                    # the second case is slightly redundant but we have some strange cases
                    if (y1 < y3) and (y2 < y4):
                        lb.add(l_index)
                        ub.add(l2_index)
                    elif (y1 > y3) and (y2 > y4):
                        lb.add(l2_index)
                        ub.add(l_index)

        lb = sorted(list(lb))
        ub = sorted(list(ub))
        for i in lb:
            if horiz:
                # lb_lines.append(multiline[i])
                lb_lines.extend(multiline[i])
            else:
                # else flip
                (x1,y1),(x2,y2) = multiline[i]
                lb_lines.extend(((y1,x1),(y2,x2)))

        for i in ub:
            if horiz:
                ub_lines.extend(multiline[i])
            else:
                # else flip
                (x1,y1),(x2,y2) = multiline[i]
                ub_lines.extend([(y1,x1),(y2,x2)])

        if True:
            if lb_lines != []:
                X,Y = zip(*lb_lines)
                plt.plot(X,Y,"-",color="red")
            if ub_lines != []:
                X,Y = zip(*ub_lines)
                plt.plot(X,Y,"-",color="blue")

    return retval

# h_lines = analysis(horiz_list,horiz_intercepts,horiz=True)
# assert False
v_lines = analysis(vert_list,vert_intercepts,horiz=False)

# ax1.set_title('Probabilistic Hough')
ax1.set_axis_off()
ax1.set_adjustable('box-forced')


# for row_index in range(len(h_lines)-1):
#     for column_index in range(len(v_lines)-1):
#         # start with the top left
#         lower_horiz = h_lines[row_index][1]
#         lower_vert = v_lines[column_index][1]
#         # print lower_horiz
#         # print lower_vert
#         multi_intersection(lower_horiz,lower_vert)
#         break
#     break
plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
plt.close()