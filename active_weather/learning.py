#!/usr/bin/env python
from __future__ import print_function
import sqlite3 as lite
import cv2
import matplotlib.pyplot as plt
import preprocessing
__author__ = 'ggdhines'

base = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"
region_bound = (559,3282,1276,2097)

con = lite.connect('/home/ggdhines/to_upload3/active.db')

cur = con.cursor()

current_subject = None
masked_image = None

for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.abcdefghijkmnopqrstuvwxyz-":
    cur.execute("select * from characters where characters = \""+c+"\" and confidence > 80")

    for r in cur.fetchall():
        print(r)
        subject_id,region,column,row,characters,confidence,lb_x,ub_x,lb_y,ub_y = r
        if subject_id != current_subject:
            current_subject = subject_id
            fname = base + subject_id + ".JPG"
            image = cv2.imread(fname)

            sub_image = image[region_bound[2]:region_bound[3],region_bound[0]:region_bound[1]]

            gray = cv2.cvtColor(sub_image,cv2.COLOR_BGR2GRAY)
            masked_image = preprocessing.__mask_lines__(gray)

            height,width,_ = sub_image.shape

        char_image = masked_image[lb_y:ub_y+1,lb_x:ub_x+1]

        plt.imshow(char_image)
        plt.show()
