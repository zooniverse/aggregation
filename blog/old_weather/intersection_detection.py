import cv2
import numpy as np
import random
import math
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

fname = "/home/ggdhines/Databases/old_weather/images/Bear-AG-29-1939-0189.JPG"

# fig = plt.figure()
# axes = fig.add_subplot(1, 1, 1)
# image_file = cbook.get_sample_data(fname)
# image = plt.imread(image_file)
# # fig, ax = plt.subplots()
# im = axes.imshow(image)
def hesse_line_reduction(line_segments):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """

    reduced_markings = []

    for line_seg in line_segments:
        x1,y1,x2,y2 = line_seg[:4]



        dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

        try:
            tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
            theta = math.atan(tan_theta)
        except ZeroDivisionError:
            theta = math.pi/2.

        reduced_markings.append([dist,theta])

        # for cases where we have lines of text, don't want to ignore those
        if len(line_seg) > 4:
            reduced_markings[-1].append(line_seg[4])

    return reduced_markings

img = cv2.imread('/home/ggdhines/Databases/old_weather/images/Bear-AG-29-1939-0189.JPG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 5
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

points = set()

y_dict = {}

for x1,y1,x2,y2 in lines[0]:
    if x2 == x1:
        m = float("inf")
    else:
        m = (y2-y1)/float(x2-x1)
        b = y1-m*x1

    if m in [float('inf'),-float('inf')]:
        y_min = min(y1,y2)
        y_max = max(y1,y2)
        p = [(x1,y) for y in range(y_min,y_max+1)]
    else:
        assert x1 < x2
        p = [(x,m*x+b) for x in range(x1,x2+1)]

    points.update(p)
    for (x,y) in p:
        if y not in y_dict:
            y_dict[y] = 1
        else:
            y_dict[y] += 1

# y_pts = [y for (x,y) in points]
# import matplotlib.pyplot as plt
# y_pts.sort()
y_pts = y_dict.items()
y_pts.sort()

for y_i in range(len(y_pts)):

    if (y_i > 0) and (y_i < (len(y_pts)-1)):
        print y_pts[y_i-1],y_pts[y_i],y_pts[y_i+1]
    elif y_i > 0:
        print y_pts[y_i-1],y_pts[y_i]
    else:
        print y_pts[y_i],y_pts[y_i+1]


assert False

l = hesse_line_reduction(lines[0])

enumerated_l = list(enumerate(l))

# enumerated_s = list(enumerate(l))
# enumerated_s.sort(key = lambda s:s[1][1])
# print enumerated_s

enumerated_l.sort(key = lambda seg:seg[1][1])

split_points = [0,]
reference_slope = enumerated_l[0][1][1]
for j,(i,(intercept,slope)) in enumerate(enumerated_l):
    if slope - reference_slope > 0.02:
        split_points.append(j)
        reference_slope = slope
split_points.append(len(enumerated_l)+1)

for split_index in range(len(split_points)-1):
    start = split_points[split_index]
    end = split_points[split_index+1]

    local_line_segments =  enumerated_l[start:end]
    local_line_segments.sort(key = lambda seg:seg[1][0])

    reference_intercept = local_line_segments[0][1][0]
    local_split_points = [0,]
    for k,(i,(intercept,slope)) in enumerate(local_line_segments):
        if intercept - reference_intercept > 8:
            local_split_points.append(k)
            reference_intercept = intercept

    local_split_points.append(len(local_line_segments)+1)

    for index in range(len(local_split_points)-1):
        start2 = local_split_points[index]
        end2 = local_split_points[index+1]
        indices,values = zip(*local_line_segments[start2:end2])

        all_intercepts,all_slopes = zip(*values)
        local_pts =  [list(lines[0][m]) for m in indices]
        if len(local_pts) <= 20:
            continue
        # X1,Y1,X2,Y2 = zip(*local_pts)

        if np.median(all_slopes) <= math.pi/4.:
            # mostly horizational line
            assert np.median(all_slopes) >= 0
            local_pts.sort(key = lambda p:p[0])
            s = []

            for p_index in range(len(local_pts)-1):
                s.append(local_pts[p_index])
                if local_pts[p_index+1][0]-local_pts[p_index][2] > 25:
                    if len(s) >= 20:
                        plt.plot([s[0][0],s[-1][2]],[s[0][1],s[-1][3]],color="red",linewidth=0.5)
                    s = []

            s.append(local_pts[-1])
            if len(s) >= 20:
                plt.plot([s[0][0],s[-1][2]],[s[0][1],s[-1][3]],color="red",linewidth=0.5)

            p1,p2 = min(local_pts,key = lambda p:p[0]),max(local_pts,key = lambda p:p[2])
            plt.plot([p1[0],p2[2]],[p1[1],p2[3]],color="red",linewidth=0.5)
        else:
            # mostly vertical lines
            p1,p2 = min(local_pts,key = lambda p:p[1]),max(local_pts,key = lambda p:p[3])
            # plt.plot([p1[0],p2[2]],[p1[1],p2[3]],color="red")




    # local_line_segments = l[start:end]
    # local_line_segments.sort(key = lambda seg:seg[0])
    #
    # intercept_split_points = []
    # for j in range(len(local_line_segments)-1):
    #     if local_line_segments[j+1][0] - local_line_segments[j][0] > 2:
    #         intercept_split_points.append(j)
    # intercept_split_points.append(len(local_line_segments))
    #
    # prev_intercept_split = -1
    # for intercept_split in intercept_split_points:
    #     if (prev_intercept_split == -1) and (intercept_split == 0):
    #         continue
    #
    #     points = local_line_segments[prev_intercept_split+1:intercept_split+1]
    #     print points
    #     prev_intercept_split = intercept_split

plt.savefig("/home/ggdhines/t.pdf",bbox_inches=0, pad_inches=0.1,dpi=1000)
plt.show()
# # normalize
# mi = min(intercepts)
# ma = max(intercepts)
# intercepts = [(i-mi)/(ma-mi) for i in intercepts]
#
# slopes = [s/math.pi for s in slopes]
#
# l = zip(intercepts,slopes)

# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imwrite('houghlines5.jpg',img)
# print np.asarray(l)
# X = np.asarray([[s,] for s in slopes])
# print X
# db = DBSCAN(eps=0.1, min_samples=3).fit(np.asarray(slopes))
# labels = db.labels_
#
# unique_labels = set(labels)
# for k in unique_labels:
#     if k == -1:
#         continue
#
#     line_segments = [l for l,lab in zip(lines[0],labels) if lab == k]
#     intercepts,slops = zip(*line_segments)
#
#     print intercepts

# labels = db.labels_
# print len(l)
# print len(set(labels))
#
# unique_labels = set(labels)
# for k in unique_labels:
#     if k == -1:
#         continue
#     line_segments = [l for l,lab in zip(lines[0],labels) if lab == k]
#
#     class_member_mask = (labels == k)
#     intercepts,slopes = zip(*X[class_member_mask])
#     # slopes = [s*math.pi for s in slopes]
#
#     intercept_diff = max(intercepts) - min(intercepts)
#     slopes_diff = max(slopes) - min(slopes)
#
#     if slopes_diff > 0:
#         print intercept_diff,slopes_diff
#         continue
#
#     if np.median(slopes) <= (math.pi/4.):
#         first = min(line_segments,key = lambda line_seg:line_seg[0])
#         last = max(line_segments,key = lambda line_seg:line_seg[2])
#     else:
#         first = min(line_segments,key = lambda line_seg:line_seg[1])
#         last = max(line_segments,key = lambda line_seg:line_seg[3])
#
#     plt.plot([first[1],last[3]],[first[0],last[2]],color="red")
#
# plt.show()