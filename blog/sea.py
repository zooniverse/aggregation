__author__ = 'ggdhines'
import matplotlib
matplotlib.use('WXAgg')
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import cv2
import numpy as np

with AggregationAPI(592,"development") as sea:
    sea.__setup__()

    postgres_cursor = sea.postgres_session.cursor()
    select = "SELECT classification_subjects.subject_id,annotations from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 607"
    postgres_cursor.execute(select)

    for subject_id in postgres_cursor.fetchall():
        subject_id = subject_id[0]

        f_name = sea.__image_setup__(subject_id)
        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(gray_image)
        plt.show()

        # (thresh, _) = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        im_bw = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)[1]

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(im_bw)
        plt.show()


        kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(1,10))

        dx = cv2.Sobel(im_bw,cv2.CV_16S,1,0)
        dx = cv2.convertScaleAbs(dx)
        cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
        cv2.imwrite("/home/ggdhines/temp.png",close)

        _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if (cv2.arcLength(cnt,True) > 250) and (h/w > 2):
                # print cv2.arcLength(cnt,True)
                print h/w
            # if h/w > 10:
                cv2.drawContours(close,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)
        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
        # closex = close.copy()
        print
        cv2.imwrite("/home/ggdhines/vert.png",close)

        kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        dy = cv2.Sobel(im_bw,cv2.CV_16S,0,2)
        dy = cv2.convertScaleAbs(dy)
        cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

        _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if w/h > 5:
                cv2.drawContours(close,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)

        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
        cv2.imwrite("/home/ggdhines/horiz.png",close)

        continue

        im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


        for ii,cnt in enumerate(contours):
            if cnt.shape[0] > 20:
                cnt = np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))
                cnt_list = cnt.tolist()
                X,Y = zip(*cnt_list)
                plt.plot(X,Y)
                # hull = ConvexHull(cnt)
                # # plt.plot(cnt[hull.vertices,0], cnt[hull.vertices,1], 'r--', lw=2)
                #
                # shapely_points = [shapely.geometry.shape({"type": "Point", "coordinates": (x,y)}) for (x,y) in zip(X,Y)]
                # concave_hull, edge_points = alpha_shape(shapely_points,alpha=0.01)
                #
                # # print edge_points
                #
                # if isinstance(concave_hull,shapely.geometry.Polygon):
                #     # plot_polygon(ax1,concave_hull)
                #     X,Y =  zip(*list(concave_hull.exterior.coords))
                #     plt.plot(X,Y)
                # else:
                #
                #     for p in concave_hull:
                #         X,Y =  zip(*list(p.exterior.coords))
                #         plt.plot(X,Y)
                # # else:
                # #     for p in concave_hull:
                # #         plot_polygon(ax1,p)
                #
                #
                # # hull_y = [Y[simplex[0]] for simplex in hull.simplices]
                # # plt.plot(hull_x,hull_y)
                # if cv2.contourArea(cnt) > 0:
                #     print cv2.contourArea(cnt)
                #     cv2.drawContours(image, contours, ii, (0,255,0), 3)

        plt.ylim((image.shape[0],0))
        plt.xlim((0,image.shape[1]))
        plt.show()
