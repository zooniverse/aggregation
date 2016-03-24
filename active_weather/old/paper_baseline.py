from __future__ import print_function
from active_weather import ActiveWeather
import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import popen
import csv

directory = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"

# min_x,max_x,min_y,max_y
region_bounds = (559,3282,1276,2097)
project = ActiveWeather()

# project.cass_db.__get_subjects__()

fname = directory+"Bear-AG-29-1940-0720.JPG"



horizontal_grid,vertical_grid = project.__get_grid_for_table__(directory,region_bounds,fname)

rows = []
columns = []

for row_index in range(len(horizontal_grid)-1):
    lb = np.min(horizontal_grid[row_index],axis=0)[1]-region_bounds[2]
    ub = np.max(horizontal_grid[row_index+1],axis=0)[1]-region_bounds[2]

    rows.append((lb,ub))
print(len(rows))

for column_index in range(len(vertical_grid)-1):
    lb = np.min(vertical_grid[column_index],axis=0)[0]-region_bounds[0]
    ub = np.max(vertical_grid[column_index+1],axis=0)[0]-region_bounds[0]

    columns.append((lb,ub))



print(fname)
image = cv2.imread(fname)

project.cass_db.__get_subjects__()

#
# plt.imshow(image)
# plt.show()

sub_image = image[region_bounds[2]:region_bounds[3],region_bounds[0]:region_bounds[1]]

height,width,_ = sub_image.shape
# plt.imshow(sub_image)
print((width,height))
cv2.imwrite("/home/ggdhines/region2.jpg",sub_image)

stream = popen("tesseract -psm 6 /home/ggdhines/region2.jpg stdout makebox")
box_results = csv.reader(stream, delimiter=' ')

transcribed = []

for c,left,top,right,bottom,_ in box_results:
    top = height - int(top)
    bottom = height - int(bottom)

    assert top > 0
    assert bottom > 0
    left = int(left)
    right = int(right)

    transcribed.append(((bottom,top,left,right),c))

    l_y = [top,top,bottom,bottom,top]
    l_x = [left,right,right,left,left]
    l = np.asarray(zip(l_x,l_y))
    # print(l)
    cv2.polylines(sub_image,[l],True,(0,255,0))

#
# plt.imshow(image)

# for (lb,ub) in rows:
#     l = np.asarray(zip([0,width],[lb,lb]))
#     cv2.polylines(sub_image,[l],True,(0,0,255))
#     l = np.asarray(zip([0,width],[ub,ub]))
#     cv2.polylines(sub_image,[l],True,(0,0,255))

# for (lb,ub) in columns:
#     l = np.asarray(zip([lb,lb],[0,height]))
#     cv2.polylines(sub_image,[l],True,(0,255,0))
#     l = np.asarray(zip([ub,ub],[0,height]))
#     cv2.polylines(sub_image,[l],True,(0,255,0))

# for h in horizontal_grid:
#     # print(h)
#     h = h-(region_bounds[0],region_bounds[2])
#     cv2.polylines(sub_image,[h],True,(0,255,255))
#     plt.plot(h[:,0]-region_bounds[0],h[:,1]-region_bounds[2])
cv2.imwrite("/home/ggdhines/test.jpg",sub_image)

transcribed_dict = {}
gold_dict = {}

for (top,bottom,left,right),t in transcribed:
    if t == None:
        continue



    in_row = False

    for row_index,(lb,ub) in enumerate(rows):
        assert top < bottom
        in_row = top>=lb and bottom <= ub
        if in_row:
            break

    if not in_row:
        continue

    in_column = False
    for column_index,(lb,ub) in enumerate(columns):
        in_column = left>=lb and right <= ub
        if in_column:
            break

    if not in_column:
        continue


    if (row_index,column_index) not in transcribed_dict:
        transcribed_dict[(row_index,column_index)] = [left],[t]
    else:
        transcribed_dict[(row_index,column_index)][0].append(left)
        transcribed_dict[(row_index,column_index)][1].append(t)

    gold = project.cass_db.__get_gold_standard__("Bear-AG-29-1940-0720",0,row_index,column_index)

    gold_dict[(row_index,column_index)] = gold

    # print(row_index,column_index)
    #
    #
    # x = np.asarray([left,left,right,right,left])
    # y = np.asarray([top,bottom,bottom,top,top])
    # print(t)
    # plt.imshow(sub_image)
    # plt.plot(x,y)
    # plt.show()

for k in transcribed_dict:
    text_with_coords = zip(transcribed_dict[k][0],transcribed_dict[k][1])
    text_with_coords.sort(key = lambda x:x[0])
    _,text_list = zip(*text_with_coords)
    text = "".join(text_list)

    print(text,gold_dict[k],text==gold_dict[k])