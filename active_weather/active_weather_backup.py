#!/usr/bin/env python
from __future__ import print_function
import matplotlib.pyplot as plt
import cv2
import preprocessing
from copy import deepcopy
import glob
import numpy as np
__author__ = 'ggdhines'

count = -1


class ActiveWeather:
    def __init__(self):
        self.regions =  [(559,3282,1276,2097)]

    def __extract_region__(self,fname,region_id = 0):
        img = cv2.imread(fname)
        region = self.regions[region_id]
        sub_image = img[region[2]:region[3],region[0]:region[1]]

        return sub_image

project = ActiveWeather()

cur = preprocessing.con.cursor()


cur.execute("create table transcriptions(subject_id text, region int, column int, row int, contents text, confidence float)")
cur.execute("create table characters(subject_id text, region int, column int, row int, characters text, confidence float,lb_x int,ub_x int, lb_y int,ub_y int)")

for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:40]:

    img = project.__extract_region__(fname)
    id_ = fname.split("/")[-1][:-4]
    print(id_)

    preprocessing.__image_run__()


    # # if not inverted:
    # best_threshold = None
    # best_confidence = 0
    # for threshold_value in range(170,191,5):
    #     pca_image,_ = preprocessing.__pca__(img,threshold_value)
    #     masked_image = preprocessing.__mask_lines__(gray,pca_image)
    #
    #     transcriptions = preprocessing.__ocr_image__(masked_image)
    #     _,confidence_values,_ = preprocessing.__place_in_cell__(transcriptions,gray,id_,save_to_db=False)
    #
    #     if np.mean(confidence_values) > best_confidence:
    #         best_threshold = threshold_value
    # pca_image,_ = preprocessing.__pca__(img,180)
    #     raw_input("enter something")
    #
    # continue




    # preprocessing.__db__(masked_image)
    # continue



    if problems > 50:
        pca_image,inverted = preprocessing.__pca__(img,180,True)


    cur = preprocessing.con.cursor()

    # cur.execute("select * from transcriptions where column = 19")
    # for fname,region_id,column,row,text,c in cur.fetchall():
    #     print(column,row,text,c)

    # assert False

    cur.execute("select * from transcriptions where confidence <= 80 and subject_id =\""+id_+"\";")
    # print(preprocessing.cur.fetchone())

    horizontal_grid,vertical_grid = preprocessing.__cell_boundaries__(gray)

    height,width,_ = img.shape

    t = 0

    with open("/home/ggdhines/to_upload3/manifest.csv","a") as csvfile:
        for fname,region_id,column,row,_,_ in cur.fetchall():
            t += 1
            count += 1
            img2 = deepcopy(img)
            y1,y2 = horizontal_grid[row],horizontal_grid[row+1]

            x1,x2 = int(vertical_grid[column]),int(vertical_grid[column+1])

            y1 = int(y1)
            y2 = int(y2)

            cv2.line(img2,(x1,y1),(x2,y1),color=(0,0,255),thickness=3)
            cv2.line(img2,(x1,y2),(x2,y2),color=(0,0,255),thickness=3)

            cv2.line(img2,(x1,y1),(x1,y2),color=(0,0,255),thickness=3)
            cv2.line(img2,(x2,y1),(x2,y2),color=(0,0,255),thickness=3)

            sub_image = img2[:,max(x1-10,0):min(x2+10,width)]

            cv2.imwrite("/home/ggdhines/to_upload3/cell"+str(count)+".jpg",sub_image)

            csvfile.write("cell"+str(count)+".jpg,"+fname+","+str(region_id)+","+str(column)+","+str(row)+"\n")
    # print(t)
        # plt.imshow(sub_image)
        # plt.show()