import numpy as np
import cv2
import csv
from os import popen
import paper_quad
import sobel_transform
import math
import matplotlib.pyplot as plt
__author__ = 'ggdhines'

image = cv2.imread("/home/ggdhines/hello_world.jpg")
gray = cv2.imread("/home/ggdhines/hello_world.jpg",0)

stream = popen("tesseract -psm 6 /home/ggdhines/hello_world.jpg stdout makebox")
# box_results = csv.reader(stream, delimiter=' ')

transcribed = []

height,width = gray.shape

# print(box_results[0])

for l in stream.readlines():
    parsed_l = l[:-1].split(" ")
    c,left,top,right,bottom,_ = parsed_l

    top = height - int(top)
    bottom = height - int(bottom)
    # print(top,bottom)
    assert top > 0
    assert bottom > 0
    left = int(left)
    right = int(right)

    print(c,(top-bottom),(right-left))

    if min((top-bottom),(right-left)) <= 10:
        l_y = [top,top,bottom,bottom,top]
        l_x = [left,right,right,left,left]
        l = np.asarray(zip(l_x,l_y))
        # print(l)
        cv2.drawContours(gray,[l],0,255,-1)
    else:
        l_y = [top,top,bottom,bottom,top]
        l_x = [left,right,right,left,left]
        l = np.asarray(zip(l_x,l_y))
        cv2.drawContours(image,[l],0,(255,0,0),1)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("/home/ggdhines/masked2.jpg",gray)

stream = popen("tesseract -psm 6 /home/ggdhines/masked2.jpg stdout makebox")

contents = stream.readlines()


transcribed = []



box_results = csv.reader(stream, delimiter=' ')

transcribed_dict = {}
gold_dict = {}

rows,columns = sobel_transform.__sobel_image__()

# for c,left,top,right,bottom,_ in box_results:
for row in contents:
    try:
        c,left,top,right,bottom,_ = row[:-1].split(" ")
    except ValueError:
        print(row)
        raise
    # print c,left,top,right,bottom
    mid_y = height-(int(top)+int(bottom))/2.
    mid_x = (int(right)+int(left))/2.

    if c == None:
        continue

    in_row = False

    for row_index in range(len(rows)-1):
        lb = rows[row_index]
        ub = rows[row_index+1]

        in_row = lb <= mid_y <= ub
        if in_row:
            break

    if not in_row:
        continue

    in_column = False
    for column_index in range(len(columns)-1):
        lb = columns[column_index]
        ub = columns[column_index+1]
        in_column = lb <= mid_x <= ub
        if in_column:
            break

    if not in_column:
        continue

    key = (row_index,column_index)
    if key not in transcribed_dict:
        transcribed_dict[key] = [(mid_x,c)]
    else:
        transcribed_dict[key].append((mid_x,c))
        # transcribed_dict[key][1].append(c)

# print transcribed_dict.keys()

total = 0
correct_empty = 0
empty = 0
# print(transcribed_dict)
with open("/home/ggdhines/gold_standard.txt","rb") as f:
    reader = csv.reader(f, delimiter=',')
    for row,column,gold_standard in reader:
        key = (int(row),int(column))

        if gold_standard == "":
            empty += 1
            if  key not in transcribed_dict:
                correct_empty += 1
        else:
            try:
                t = sorted(transcribed_dict[key],key = lambda x:x[0])

            except KeyError:
                # print("skipping")
                continue
            _,chrs = zip(*t)
            text = "".join(chrs)
            # print (row,column),gold_standard,text
            if gold_standard == text:
                total += 1
            else:
                print((gold_standard,text))
            # if gold_standard == None and text == "":
            #     correct_empty += 1
            # if gold_standard == None:
            #     print text
            #     assert Fase
            #     empty += 1

print(total)
print(correct_empty)
print(empty)