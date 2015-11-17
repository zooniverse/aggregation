from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt

import math
from skimage.transform import hough_line,hough_line_peaks


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

# def deter(a,b,c,d):
#     return a*d - c*b
#
# def intersection(h_l,v_l):
#     (x1,y1),(x2,y2) = h_l
#
#     (x3,y3),(x4,y4) = v_l
#
#     d1 = deter(x1,y1,x2,y2)
#     d2 = deter(x1,1,x2,1)
#     d3 = deter(x3,y3,x4,y4)
#     d4 = deter(x3,1,x4,1)
#
#     D1 = deter(d1,d2,d3,d4)
#
#     d5 = deter(y1,1,y2,1)
#     d6 = deter(y3,1,y4,1)
#
#     D2 = deter(d1,d5,d3,d6)
#
#     d7 = deter(x3,1,x4,1)
#     d8 = deter(y3,1,y4,1)
#
#     D3 = deter(d2,d5,d7,d8)
#
#     intersect_x = int(D1/D3)
#     intersect_y = int(D2/D3)
#
#     return intersect_x,intersect_y

img = cv2.imread("/home/ggdhines/Databases/old_weather/cells/Bear-AG-29-1939-0187_1_7.png")

print img.shape

colours = {}

# under the assumption that most of the cell is not ink - find the most common pixel colour
# any pixel that is "far" enough away is assumed to be ink
for c in range(img.shape[1]):
    for r in range(img.shape[0]):
        pixel_colour = tuple(img[r,c])

        if pixel_colour not in colours:
            colours[pixel_colour] = 1
        else:
            colours[pixel_colour] += 1

most_common_colour,_ = sorted(colours.items(),key = lambda x:x[1],reverse=True)[0]

image = np.zeros(img.shape[:2])
for c in range(img.shape[1]):
    for r in range(img.shape[0]):
        pixel_colour = tuple(img[r,c])
        dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(pixel_colour,most_common_colour)]))

        if dist > 40:
            image[(r,c)] = 150

# image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



h, theta, d = hough_line(image)



# for i in h:
#     print len(i)
# assert False

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_axis_off()

# ax2.imshow(np.log(1 + h),
#     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
#          d[-1], d[0]],
#     cmap=plt.cm.gray, aspect=1/1.5)
# ax2.set_title('Hough transform')
# ax2.set_xlabel('Angles (degrees)')
# ax2.set_ylabel('Distance (pixels)')
# ax2.axis('image')

lb_lines = []
ub_lines = []
lhs_lines = []
rhs_lines = []

ax2.imshow(image)#, cmap=plt.cm.gray)
rows, cols = image.shape
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
print image.shape
print 190/float((image.shape[1]))
for d_index in range(len(d)):
    for theta_index in range(len(theta)):
        votes = h[d_index][theta_index]
        # if votes > 0:
        #     print votes

        angle = theta[theta_index]
        t1 = min(math.fabs(angle-math.pi),math.fabs(angle))
        t2 = math.fabs(angle-math.pi/2.)
        dist = d[d_index]

        # vertical
        if (t1 < 0.05) and (votes /float((image.shape[0])) >= 0.8):
            # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            # y1 = (dist - cols * np.cos(angle)) / np.sin(angle)

            x0 = (dist - 0 * np.sin(angle)) / np.cos(angle)
            x1 = (dist - rows * np.sin(angle)) / np.cos(angle)

            if (x0+x1)/2. <= (cols/2.):
                # lhs_lines.append(((x0,x1),(0,rows)))
                lhs_lines.append(line((x0,0),(x1,rows)))
                ax2.plot((x0, x1), (0, rows), '-r')
            else:
                # rhs_lines.append(((x0,x1),(0,rows)))
                rhs_lines.append(line((x0,0),(x1,rows)))

        elif (t2 < 0.05) and (votes /float((image.shape[1])) >= 0.8):
            print angle,t2

            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
            # ax2.plot((0, cols), (y0, y1), '-b')

            # ax2.plot((x0, x1), (0, rows), '-b')

            if (y0+y1)/2. <= (rows/2.):
                ax2.plot((0, cols), (y0, y1), '-b')
                # lb_lines.append(((0,cols),(y0,y1)))
                lb_lines.append(line((0,y0),(cols,y1)))
            else:
                ub_lines.append(line((0,y0),(cols,y1)))

ax2.axis((0, cols, rows, 0))
ax2.set_title('Detected lines')
ax2.set_axis_off()

ax3.imshow(image)

lr_corners = []

for l1 in lb_lines:
    for l2 in rhs_lines:
        # print l1
        # print l2
        # x,y = intersection(l1,l2)
        lr_corners.append(intersection(l1,l2))
        # ax3.plot(x,y,'o',color="green")

X,Y = zip(*lr_corners)
#ax3.plot(min(X),max(Y),'o',color="green")
lr_x,lr_y = min(X),max(Y)

ur_corners = []

for l1 in ub_lines:
    for l2 in rhs_lines:
        # print l1
        # print l2
        # x,y = intersection(l1,l2)
        ur_corners.append(intersection(l1,l2))
        # ax3.plot(x,y,'o',color="green")

X,Y = zip(*ur_corners)
# ax3.plot(min(X),min(Y),'o',color="green")
ur_x,ur_y = min(X),min(Y)

ul_corners = []

for l1 in ub_lines:
    for l2 in lhs_lines:
        # print l1
        # print l2
        # x,y = intersection(l1,l2)
        ul_corners.append(intersection(l1,l2))
        # ax3.plot(x,y,'o',color="green")

X,Y = zip(*ul_corners)
# ax3.plot(max(X),min(Y),'o',color="green")
ul_x,ul_y = max(X),min(Y)

ll_corners = []

for l1 in lb_lines:
    for l2 in lhs_lines:
        # print l1
        # print l2
        # x,y = intersection(l1,l2)
        ll_corners.append(intersection(l1,l2))
        # ax3.plot(x,y,'o',color="green")

X,Y = zip(*ll_corners)
# ax3.plot(max(X),max(Y),'o',color="green")
ll_x,ll_y = max(X),max(Y)

ax3.plot([ll_x,lr_x],[ll_y,lr_y],"-g")
ax3.plot([ll_x,ul_x],[ll_y,ul_y],"-g")
ax3.plot([ul_x,ur_x],[ul_y,ur_y],"-g")
ax3.plot([ur_x,lr_x],[ur_y,lr_y],"-g")

ax3.axis((0, cols, rows, 0))
ax3.set_title('Detected lines')
ax3.set_axis_off()

plt.show()