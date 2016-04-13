from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
from aggregation_api import AggregationAPI
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import cv2
import numpy as np
import math
from skimage.feature import blob_dog, blob_log, blob_doh

butterflies = AggregationAPI(1150,"quasi")
butterflies.__setup__()

def closest(aim,points):
    distance = float("inf")
    best_point = None

    for p in points:
        d = math.sqrt((aim[0]-p[0])**2+(aim[1]-p[1])**2)

        if d < distance:
            distance = d
            best_point = p

    return tuple(best_point)

# fname = butterflies.__image_setup__(1120709)[0]
for subject_id in butterflies.__get_subjects_in_workflow__(874):
    fname = butterflies.__image_setup__(subject_id)[0]
    print(fname)

    image_file = cbook.get_sample_data(fname)
    image = plt.imread(image_file)
    plt.imshow(image)
    plt.show()

    image = np.asarray(image,dtype=np.int)

    d1 = np.power(image[:,:,0] - 211,2)
    d2 = np.power(image[:,:,1] - 44,2)
    d3 = np.power(image[:,:,2] - 124,2)

    overall_d = np.power(d1+d2+d3,0.5)
    overall_d = np.uint8(255 - cv2.normalize(overall_d,overall_d,0,255,cv2.NORM_MINMAX))

    second_template = overall_d[:]

    # second_template[np.where(second_template > 100)] = 255

    ret,second_template = cv2.threshold(overall_d,100,255,cv2.THRESH_BINARY)

    plt.imshow(second_template,cmap="gray")
    plt.show()

    third_template = np.zeros(image.shape[:2],np.uint8)
    third_template.fill(255)

    _,contour, hier = cv2.findContours(second_template.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt,h in zip(contour,hier[0]):
        if 50 < cv2.arcLength(cnt,True) < 1000:
            x,y,w,h = cv2.boundingRect(cnt)

            approx_area = cv2.minEnclosingCircle(cnt)[1]**2 * math.pi
            actual_area = cv2.contourArea(cnt)
            if actual_area == 0:
                continue

            approx_radius = 2*math.pi*math.sqrt(actual_area/math.pi)

            if (approx_area/actual_area < 2.5) and (approx_radius/cv2.arcLength(cnt,True) > 0.5):



                print(approx_area/actual_area,approx_radius/cv2.arcLength(cnt,True))


                # print(cv2.contourArea(cnt)/cv2.arcLength(cnt,True),(w/float(h)))
                # print(cv2.arcLength(cnt,True))
                cv2.drawContours(third_template,[cnt],0,0,1)

    plt.imshow(third_template,cmap="gray")
    plt.show()
    continue

    # plt.show()

    ret,overall_d = cv2.threshold(overall_d,200,255,cv2.THRESH_BINARY)
    # overall_d[np.where(overall_d > 200)] = 255

    plt.imshow(second_template,cmap="gray")
    plt.show()

    blobs_log =blob_doh(second_template, max_sigma=30, threshold=.01)
    # blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)
    # fig,axes = plt.subplots(1, 1)
    # axes[0].imshow(second_template,cmap="gray")

    image = np.asarray(image,dtype=np.uint8)
    for b in blobs_log:
        y, x, r = b
        if 5 < r < 20:
            print(r)
            # c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            cv2.circle(image,(x,y),r,(255,255,255))
    plt.imshow(image,cmap="gray")
    plt.show()



    template = np.zeros(image.shape[:2],np.uint8)
    template.fill(255)

    _,contour, hier = cv2.findContours(overall_d.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt,h in zip(contour,hier[0]):
        x,y,w,h_ = cv2.boundingRect(cnt)
        if 0.5 <= (h_/float(w)) <= 2:
            if 200 <= cv2.arcLength(cnt,True) < 2000:

                print(cv2.arcLength(cnt,True))
                cv2.drawContours(template,[cnt],0,0,1)

                left_most_point = tuple(cnt[cnt[:,:,0].argmin()][0])
                top_point = tuple(cnt[cnt[:,:,1].argmin()][0])

                right_most_point = tuple(cnt[cnt[:,:,0].argmax()][0])
                bottom_point = tuple(cnt[cnt[:,:,1].argmax()][0])

                cnt = np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))

                extreme_point_1 = (left_most_point[0],top_point[1])
                closest_1 = closest(extreme_point_1,cnt)

                extreme_point_2 = (right_most_point[0],bottom_point[1])
                closest_2 = closest(extreme_point_2,cnt)

                cv2.line(template,closest_1,closest_2,0)

                # cv2.line(template,tuple(left_most_point),extreme_point,0)
                # cv2.line(template,extreme_point,tuple(top_point),0)

                continue

                right_most_point = cnt[:,:,0].argmax()
                top_point = cnt[:,:,1].argmin()
                bottom_point =cnt [:,:,1].argmax()

                max_distance = 0
                best_i = None
                best_j = None
                for i in range(cnt.shape[0]):
                    p1 = cnt[i]
                    for j in range(cnt.shape[1]):
                        p2 = cnt[j]

                        dist = math.sqrt((p1[0][0]-p2[0][0])**2+(p1[0][1]-p2[0][1])**2)

                        if dist > max_distance:
                            max_distance = dist
                            best_i = i
                            best_j = j

                print(cnt[best_i][0],cnt[best_j][0])
                cv2.line(template,tuple(cnt[best_i][0]),tuple(cnt[best_j][0]),0)


                #hull = cv2.convexHull(cnt,clockwise=True)




    # close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
    # closey = close.copy()
    plt.imshow(template,cmap="gray")
    plt.show()

    continue


    res = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    # dy = cv2.Sobel(res,cv2.CV_16S,0,2)
    # dy = cv2.convertScaleAbs(dy)
    # cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    # ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #
    # t= image
    # _,contour, hier = cv2.findContours(close.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contour:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     if ((w/h)>5) and (w>130) and (w < 160):
    #         print(w)
    #         cv2.drawContours(t,[cnt],0,(0,255,0),-1)
    #
    # im = plt.imshow(t)
    # plt.show()


    edges = cv2.Canny(res,50,100)
    plt.imshow(edges,cmap="gray")
    plt.show()

# th3 = cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,351,2)
# plt.imshow(th3,cmap="gray")
# plt.show()
#
# ret2,th2 = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(th2,cmap="gray")
# plt.show()
#
# _,contour, hier = cv2.findContours(res.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# t = np.zeros(res.shape,np.uint8)
# for cnt,h in zip(contour,hier):
#     print(h)
#     cv2.drawContours(t,[cnt],0,255,-1)
#
# plt.imshow(t,cmap="gray")
# plt.show()
# #     x,y,w,h = cv2.boundingRect(cnt)