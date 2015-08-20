import cv2
import numpy as np

from scipy import spatial

minVal = 25
maxVal = 100

img = cv2.imread("/home/ggdhines/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,minVal,maxVal,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,350)




img = cv2.imread("/home/ggdhines/Dropbox/789c61ed-84b5-4f8b-b372-a244889f6588.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,minVal,maxVal,apertureSize = 3)

lines_2 = cv2.HoughLines(edges,1,np.pi/180,350)


intercepts,slopes = zip(*lines[0])
intercepts_2,slopes_2 = zip(*lines_2[0])

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

mapping_to_1 = [[] for i in lines[0]]
mapping_to_2 = [[] for i in lines_2[0]]

for ii,x in enumerate(zip(normalized_i_2,normalized_s_2)):
    dist,neighbour = tree.query(x)
    # print dist,neighbour
    mapping_to_1[neighbour].append((ii,dist))

for ii,x in enumerate(zip(normalized_i,normalized_s)):
    dist,neighbour = tree_2.query(x)
    mapping_to_2[neighbour].append((ii,dist))

# print mapping_to_1
# print mapping_to_2

to_draw_1 = []
to_draw_2 = []

for i in range(len(lines[0])):
    # find a bijection
    # so line[0][i] is the closest line to line_2[0][j], make sure that
    # line_2[0][j] is also the closest line to line[0][i]
    # if such a bijection does not exist, ignore this line
    # bijection = None
    for j,dist in mapping_to_1[i]:
        for i_temp,dist_2 in mapping_to_2[j]:
            if i_temp == i:
                if max(dist,dist_2) < 0.001:
                    # bijection = j
                    to_draw_1.append(lines[0][i])
                    to_draw_2.append(lines_2[0][j])
                    # print lines[0][i]
                    # print lines_2[0][j]
                    # print
                    print max(dist,dist_2)

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

img = cv2.imread("/home/ggdhines/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
for rho,theta in to_draw_1:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('/home/ggdhines/houghlines3.jpg',img)

print len(to_draw_1)
print len(to_draw_2)

img = cv2.imread("/home/ggdhines/Dropbox/789c61ed-84b5-4f8b-b372-a244889f6588.jpeg")
for rho,theta in to_draw_2:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('/home/ggdhines/houghlines1.jpg',img)