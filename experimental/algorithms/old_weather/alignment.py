import cv2
import numpy as np
import os
from scipy import spatial
import math


def deter(a,b,c,d):
    return a*d - c*b

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

minVal = 25
maxVal = 100

img = cv2.imread(base_directory + "/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,minVal,maxVal,apertureSize = 3)

lines_t = cv2.HoughLines(edges,1,np.pi/180,350)

lines = [l[0] for l in lines_t]
# for l in lines_t:
#     print l[0]

# print len(lines)
# print lines[0]
# assert False

img = cv2.imread(base_directory + "/Dropbox/789c61ed-84b5-4f8b-b372-a244889f6588.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,minVal,maxVal,apertureSize = 3)

lines_2_t = cv2.HoughLines(edges,1,np.pi/180,350)
lines_2 = [l[0] for l in lines_2_t]

# intercepts,slopes = zip(*lines[0])
# intercepts_2,slopes_2 = zip(*lines_2[0])

intercepts,slopes = zip(*lines)
intercepts_2,slopes_2 = zip(*lines_2)

print intercepts
print slopes

max_i = max(max(intercepts),max(intercepts_2))
min_i = min(min(intercepts),min(intercepts_2))
max_s = max(max(slopes),max(slopes_2))
min_s = min(min(slopes),min(slopes_2))

normalized_s = [(s-min_s)/(max_s-min_s) for s in slopes]
normalized_i = [(i-min_i)/(max_i-min_i) for i in intercepts]

normalized_s_2 = [(s-min_s)/(max_s-min_s) for s in slopes_2]
normalized_i_2 = [(i-min_i)/(max_i-min_i) for i in intercepts_2]

tree = spatial.KDTree(zip(normalized_i,normalized_s))
tree_2 = spatial.KDTree(zip(normalized_i_2,normalized_s_2))

mapping_to_1 = [[] for i in lines]
mapping_to_2 = [[] for i in lines_2]

for ii,x in enumerate(zip(normalized_i_2,normalized_s_2)):
    dist,neighbour = tree.query(x)
    # print dist,neighbour
    # print neighbour
    # print lines[0]
    mapping_to_1[neighbour].append((ii,dist))

for ii,x in enumerate(zip(normalized_i,normalized_s)):
    dist,neighbour = tree_2.query(x)
    mapping_to_2[neighbour].append((ii,dist))

# print mapping_to_1
# print mapping_to_2

to_draw_1 = []
to_draw_2 = []

for i in range(len(lines)):
    # find a bijection
    # so line[0][i] is the closest line to line_2[0][j], make sure that
    # line_2[0][j] is also the closest line to line[0][i]
    # if such a bijection does not exist, ignore this line
    # bijection = None
    for j,dist in mapping_to_1[i]:
        for i_temp,dist_2 in mapping_to_2[j]:
            if i_temp == i:
                # print max(dist,dist_2)
                if max(dist,dist_2) < 0.001:
                    to_draw_1.append(lines[i])
                    to_draw_2.append(lines_2[j])
                    # print lines[0][i]
                    # print lines_2[0][j]
                    # print
                    # print max(dist,dist_2)

                break

    # bijection_l = [j for j in mapping_to_1[i] if (i in mapping_to_2[j])]
    # # for j in mapping_to_1[i]:
    # #     print i in mapping_to_2[j]
    # # print
    # # print bijection_l
    # # there is a bijection
    # if bijection_l != []:
    #     bijection = bijection_l[0]
    #
    #     to_draw_1.append(lines[0][i])
    #     to_draw_2.append(lines_2[0][bijection])
    #
    #     print lines[0][i]
    #     print lines_2[0][bijection]
    #     print
# assert False

bijections = zip(to_draw_1,to_draw_2)
for a,b in bijections:
    print a,b

# img = cv2.imread(base_directory + "/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
# for rho,theta in to_draw_1:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite(base_directory + '/houghlines3.jpg',img)
#
# print len(to_draw_1)
# print len(to_draw_2)
#
# img = cv2.imread(base_directory + "/Dropbox/789c61ed-84b5-4f8b-b372-a244889f6588.jpeg")
# for rho,theta in to_draw_2:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# cv2.imwrite(base_directory + '/houghlines1.jpg',img)


def intersections(lines):
    line_intersections = {}

    for l1_index,l1 in enumerate(lines):
        if l1[0] < 0:
            l1 = -l1[0],(l1[1]+math.pi/2.)%math.pi
        for l2_index,l2 in enumerate(lines[:l1_index+1]):
            l2_index += l1_index+1
            if l2[0] < 0:
                l2 = -l2[0],(l2[1]+math.pi/2.)%math.pi
            angle_diff = math.fabs(min(l1[1]-l2[1], math.pi-l1[1]-l2[1]))
            if angle_diff > 0.1:
                # print angle_diff
                # print l1
                # print l2
                # print

                x1 = math.cos(l1[1])*l1[0]
                y1 = math.sin(l1[1])*l1[0]

                y2 = None
                x2 = None

                theta_1 = l1[1]

                # vertical line
                if math.fabs(theta_1%math.pi) < 0.01:
                    x2 = x1
                    y2 = 10
                # horizontal line
                elif math.fabs(theta_1%(math.pi/2.)) < 0.01:
                    x2 = 10
                    y2 = y1
                # elif

                # print (x1,y1),(x2,y2)

                # see https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
                # for logic and notation
                x3 = math.cos(l2[1])*l2[0]
                y3 = math.sin(l2[1])*l2[0]

                x4 = None
                y4 = None

                theta_2 = l2[1]

                # vertical line
                if math.fabs(theta_2%math.pi) < 0.01:
                    x4 = x3
                    y4 = 15
                # horizontal
                elif math.fabs(theta_2%(math.pi/2.)) < 0.01:
                    x4 = 15
                    y4 = y3

                if None in [x1,x2,x3,x4,y1,y2,y3,y4]:
                    # print "skipping"
                    continue

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

                # print intersect_x,intersect_y

                line_intersections[(l1_index,l2_index)] = (intersect_x,intersect_y)

            # cv2.circle(img,(intersect_x,intersect_y),20,(0,0,255))
    return line_intersections
pts1 = intersections(to_draw_1)
pts2 = intersections(to_draw_2)

x_displacements = []
y_displacements = []

for line_tuple in pts1.keys():
    if line_tuple in pts2:
        d_x = pts2[line_tuple][0] - pts1[line_tuple][0]
        d_y = pts2[line_tuple][1] - pts1[line_tuple][1]

        x_displacements.append(d_x)
        y_displacements.append(d_y)

print min(x_displacements),np.mean(x_displacements),max(x_displacements)
print min(y_displacements),np.mean(y_displacements),max(y_displacements)

# cv2.imwrite(base_directory + '/houghlines3.jpg',img)