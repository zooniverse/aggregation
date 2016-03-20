# from __future__ import print_function
# from active_weather import ActiveWeather
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sobel_transform
import csv

directory = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"

# min_x,max_x,min_y,max_y
region_bounds = (559,3282,1276,2097)
# project = ActiveWeather()

# project.cass_db.__get_subjects__()

fname = directory+"Bear-AG-29-1940-0720.JPG"

rows,columns = sobel_transform.__sobel_image__()

table = cv2.imread("/home/ggdhines/sobel_masked.jpg",0)
print table.size

already_done = []

with open("/home/ggdhines/gold_standard.txt","r") as f:
    reader = csv.reader(f, delimiter=',')
    for row,column,_ in reader:
        already_done.append((int(row),int(column)))

with open("/home/ggdhines/gold_standard.txt","a") as f:

    for row_index in range(len(rows)-1):
        row_lb = rows[row_index]
        row_ub = rows[row_index+1]
        for column_index in range(len(columns)-1):
            if (row_index,column_index) in already_done:
                continue

            column_lb = columns[column_index]
            column_ub = columns[column_index+1]
            for r in range(int(row_lb),int(row_ub)+1):
                # print (column_lb,column_ub)

                for pixel in table[r][column_lb:column_ub+1]:
                    if pixel == 255:
                        print "\b ",
                    else:
                        print "\b*",
                print
            print (row_index,column_index)
            cell = raw_input("enter cell points: ")

            f.write(str(row_index) + "," + str(column_index) + "," + cell + "\n")