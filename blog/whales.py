__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import cv2
import numpy as np

with AggregationAPI(11,"development") as whales:
    whales.__setup__()

    postgres_cursor = whales.postgres_session.cursor()
    select = "SELECT classification_subjects.subject_id,annotations from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84"
    postgres_cursor.execute(select)

    for subject_id,annotations in postgres_cursor.fetchall():
        f_name = whales.__image_setup__(subject_id)

        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)
        plt.show()

        inds_0 = image[:,:,0] >= 100
        inds_1 = image[:,:,1] >= 100
        inds_2 = image[:,:,2] >= 100
        inds_white = inds_0 & inds_1 & inds_2

        inds = image[:,:,2] >= 50
        image[inds] = [255,255,255]
        image[inds_white] = [0,0,0]


        # imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # ret,thresh = cv2.threshold(imgray,127,255,0)
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)
        plt.show()

        continue

        fig, ax1 = plt.subplots(1, 1)

        edges = cv2.Canny(image,50,400)

        im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for ii,cnt in enumerate(contours):
            cnt = np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))
            cnt_list = cnt.tolist()
            X,Y = zip(*cnt_list)
            plt.plot(X,Y)
            # if cv2.contourArea(cnt) > 0:
            #     print cv2.contourArea(cnt)
            #     cv2.drawContours(image, contours, ii, (0,255,0), 3)

        plt.ylim((image.shape[0],0))
        plt.xlim((0,image.shape[0]))
        plt.show()
