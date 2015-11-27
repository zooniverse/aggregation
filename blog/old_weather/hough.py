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
edges = cv2.Canny(gray,50,150,apertureSize = 3)
# cv2.imwrite('/home/ggdhines/1.jpg',edges)

lines = probabilistic_hough_line(edges, threshold=10, line_length=50,line_gap=0)
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(52,78)
ax1.imshow(image)

horiz_list = []
horiz_intercepts = []

vert_list = []
vert_intercepts = []

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

    X = np.asarray(normalized_d)
    # print X
    db = DBSCAN(eps=0.005, min_samples=3).fit(X)
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
            line = lines[i]
            p0, p1 = line
            X = p0[0],p1[0]
            Y = p0[1],p1[1]
            ax1.plot(X, Y,color=col)

            # if horiz, sort by increasing X values
            if horiz:
                if p0[0] < p1[0]:
                    multiline.append(line)
                else:

                    multiline.append((p1,p0))
            else:
                # sort y increasing Y values
                if p0[1] < p1[1]:
                    multiline.append(line)
                else:
                    multiline.append((p1,p0))

        multiline.sort(key = lambda x:x[0][0])

        lb = set()
        ub = set()

        for l_index,line in enumerate(multiline):
            for l2_lindex,line_2 in list(enumerate(multiline))[l_index+1:]:
                # if the starting point of the next line segment is after the ending point of the current line segment
                # stop
                if horiz:
                    if line_2[0][0] > line[1][0]:
                        break

                    if line_2[0][1] > line[0][1]:
                        ub.add(l2_lindex)
                        lb.add(l_index)
                    else:
                        lb.add(l2_lindex)
                        ub.add(l_index)
                else:
                    if line_2[0][1] > line[1][1]:
                        break

                    if line_2[0][0] > line[0][0]:
                        ub.add(l2_lindex)
                        lb.add(l_index)
                    else:
                        lb.add(l2_lindex)
                        ub.add(l_index)

        lb = sorted(list(lb))
        for ii in range(len(lb)-1):
            l_index = lb[ii]
            l2_index = lb[ii+1]
            X = multiline[l_index][1][0],multiline[l2_index][0][0]
            Y = multiline[l_index][1][1],multiline[l2_index][0][1]
            ax1.plot(X, Y,color=col)

        ub = sorted(list(ub))
        for ii in range(len(ub)-1):
            l_index = ub[ii]
            l2_index = ub[ii+1]
            X = multiline[l_index][1][0],multiline[l2_index][0][0]
            Y = multiline[l_index][1][1],multiline[l2_index][0][1]
            ax1.plot(X, Y,color=col)

        retval.append((lb,ub))


h_lines = analysis(horiz_list,horiz_intercepts,horiz=True)
v_lines = analysis(vert_list,vert_intercepts,horiz=False)


# ax1.set_title('Probabilistic Hough')
ax1.set_axis_off()
ax1.set_adjustable('box-forced')
plt.savefig("/home/ggdhines/Databases/example.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
plt.close()
#
# h, theta, d = hough_line(edges)
# fig, ax1 = plt.subplots(1, 1)
# # fig.set_size_inches(52,78)
# rows, cols,_ = image.shape
# ax1.imshow(image)
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     print angle
#     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#     y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
#     ax1.plot((0, cols), (y0, y1), '-r')
# plt.show()

# print type(img)
# print type(image)
# print img.shape,image.shape
# # edges = canny(image, 2, 1, 25)
# _,image = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# print image
#
# pts = []
#
# for x in range(image.shape[0]):
#     for y in range(image.shape[1]):
#         if image[x][y] > 0:
#             pts.append((x,y))
#
#
# tree = KDTree(np.asarray(pts))
# # lines = probabilistic_hough_line(image, threshold=10, line_length=150,line_gap=0)