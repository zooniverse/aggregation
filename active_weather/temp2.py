import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt

horizontal = []

def __get_lines__(horizontal):
    lined_images = []

    for f in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:5]:
        img = cv2.imread(f,0)

        if horizontal:
            dy = cv2.Sobel(img,cv2.CV_16S,0,2)
            dy = cv2.convertScaleAbs(dy)
            cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        else:
            dx = cv2.Sobel(img,cv2.CV_16S,2,0)
            dx = cv2.convertScaleAbs(dx)
            cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernel)

        lined_images.append(close)

    average_image = np.percentile(lined_images,40,axis=0)

    average_image = average_image.astype(np.uint8)

    cv2.imwrite("/home/ggdhines/results.jpg",average_image)

    contours_to_return = []

    _,contour, hier = cv2.findContours(average_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if (horizontal and w/h > 5) or ((not horizontal) and h/w > 5):
            cv2.drawContours(average_image,[cnt],0,255,-1)
            contours_to_return.append(cnt)
        else:
            cv2.drawContours(average_image,[cnt],0,0,-1)

    average_image = cv2.morphologyEx(average_image,cv2.MORPH_DILATE,None,iterations = 2)

    return average_image,contours_to_return

def __get_masks__():

    horizontal_image,horizontal_contours = __get_lines__(True)
    cv2.imwrite("/home/ggdhines/horizontal.jpg",horizontal_image)

    vertical_image,vertical_contours = __get_lines__(False)
    cv2.imwrite("/home/ggdhines/vertical.jpg",vertical_image)

    grid = np.max([horizontal_image,vertical_image],axis=0)
    cv2.imwrite("/home/ggdhines/grid.jpg",grid)

    _,contours, hier = cv2.findContours(grid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(grid, contours, -1, 255, 3)

    # _,contours, hier = cv2.findContours(grid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print hier
    # masks = np.zeros(grid.shape,np.uint8)

    # cv2.drawContours(masks, contours, -1, 0, 3)
    #
    # print hier
    #
    # for cnt in contour:
    #     cv2.drawContours(masks,[cnt],0,255,1)




    masks = np.zeros(grid.shape,np.uint8)

    for cnt in horizontal_contours:
        shape = cnt.shape
        t = cnt.reshape((shape[0],shape[2]))
        max_x,max_y = np.max(t,axis=0)
        min_x,min_y = np.min(t,axis=0)

        if (min_x >= 563) and (max_x <= 3282) and (min_y>=3541) and (max_y<=4346):
            perimeter = cv2.arcLength(cnt,True)
            if perimeter > 1000:
                cv2.drawContours(masks,[cnt],0,255,3)

    for cnt in vertical_contours:
        shape = cnt.shape
        t = cnt.reshape((shape[0],shape[2]))

        max_x,max_y = np.max(t,axis=0)
        min_x,min_y = np.min(t,axis=0)

        if (min_x >= 563) and (max_x <= 3282) and (max_y>=3541) and (min_y<=4346):
            perimeter = cv2.arcLength(cnt,True)
            if perimeter > 1000:
                cv2.drawContours(masks,[cnt],0,255,3)

    _,contours, hier = cv2.findContours(masks.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    masks2 = np.zeros(grid.shape,np.uint8)

    for cnt,h in zip(contours,hier[0]):
        if h[-1] == -1:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        if min(h,w) > 30:
            print cv2.arcLength(cnt,True)
            cv2.drawContours(masks2,[cnt],0,255,-1)
    cv2.imwrite("/home/ggdhines/masks.jpg",masks2)



    from sklearn.cluster import DBSCAN

    # X = np.asarray(zip(pts[0],pts[1]))
    #
    # print np.where(X[:,]>= 563)
    # print np.where(X[:,] <= 3282)


    # db = DBSCAN(eps=2, min_samples=3).fit(X)
    # labels = db.labels_
    #
    # unique_labels = set(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    #
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = 'k'
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = X[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], '.')
    #
    # plt.show()
