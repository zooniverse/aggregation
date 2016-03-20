import numpy as np
import cv2
import csv
from os import popen
import paper_quad
import sobel_transform
import math
import matplotlib.pyplot as plt
__author__ = 'ggdhines'

def __upper_bounds__(line):
    sorted_l = sorted(line, key = lambda l:l[0])

    x_ret = []
    y_ret = []

    current_x = sorted_l[0][0]
    current_y = -float("inf")

    for x,y in sorted_l:
        if x != current_x:
            x_ret.append(current_x)
            y_ret.append(current_y)
            current_x = x
            current_y = -float("inf")
        current_y = max(current_y,y)

    x_ret.append(current_x)
    y_ret.append(current_y)

    return x_ret,y_ret

def __lower_bounds__(line):
    sorted_l = sorted(line, key = lambda l:l[0])

    x_ret = []
    y_ret = []

    current_x = sorted_l[0][0]
    current_y = float("inf")

    for x,y in sorted_l:
        if x != current_x:
            x_ret.append(current_x)
            y_ret.append(current_y)
            current_x = x
            current_y = float("inf")
        current_y = min(current_y,y)

    x_ret.append(current_x)
    y_ret.append(current_y)

    return x_ret,y_ret

def __median_kernel__(img,line,horizontal):
    if horizontal:
        x,y = __upper_bounds__(line)
        x2,y2 = __lower_bounds__(line)
    else:
        y,x = zip(*line)
        line = zip(y,x)
        x,y = __upper_bounds__(line)
        x2,y2 = __lower_bounds__(line)

    x.extend(x2)
    y.extend(y2)

    y_t2 = []
    step = 60
    for y_index in range(len(y)):
        m = np.median(y[max(y_index-step,0):y_index+step])
        assert not math.isnan(m)

        y_t2.append(int(round(m)))

    mask = np.zeros(img.shape,np.uint8)

    if horizontal:
        pts = np.asarray(zip(x,y_t2))
    else:
        pts = np.asarray(zip(y_t2,x))

    cv2.drawContours(mask,[pts],0,255,-1)

    return mask

def __correct__(img,line,horizontal,background=0,foreground=255):
    assert len(img.shape) == 2
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask,[line],0,255,-1)


    mask2 = __median_kernel__(img,line,horizontal)

    # plt.imshow(mask)
    # plt.show(0)

    overlap = np.min([mask,mask2],axis=0)

    plt.imshow(overlap)
    plt.show(0)

    return overlap

if __name__ == "__main__":
    img = cv2.imread('/home/ggdhines/region.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    horizontal_lines = paper_quad.__extract_grids__(gray,True)

    mask = np.zeros(gray.shape,np.uint8)
    for l in horizontal_lines:
        corrected_l = __correct__(gray,l,True)
        mask = np.max([mask,corrected_l],axis=0)

    cv2.imwrite("/home/ggdhines/testing.jpg",mask)
    vertical_lines = paper_quad.__extract_grids__(gray,False)
    for l in vertical_lines:
        corrected_l = __correct__(gray,l,False)
        mask = np.max([mask,corrected_l],axis=0)



    ret,thresh1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
    masked_image = np.max([thresh1,mask],axis=0)

    cv2.imwrite("/home/ggdhines/masked.jpg",masked_image)

    #######
    ######

    stream = popen("tesseract -psm 6 /home/ggdhines/masked.jpg stdout makebox")
    box_results = csv.reader(stream, delimiter=' ')

    transcribed = []

    height,width = gray.shape

    for c,left,top,right,bottom,_ in box_results:
        top = height - int(top)
        bottom = height - int(bottom)
        # print(top,bottom)
        assert top > 0
        assert bottom > 0
        left = int(left)
        right = int(right)


        if min((top-bottom),(right-left)) <= 6:
            l_y = [top,top,bottom,bottom,top]
            l_x = [left,right,right,left,left]
            l = np.asarray(zip(l_x,l_y))
            # print(l)
            cv2.drawContours(masked_image,[l],0,255,-1)

    cv2.imwrite("/home/ggdhines/masked2.jpg",masked_image)

    stream = popen("tesseract -psm 6 /home/ggdhines/masked2.jpg stdout makebox")
    box_results = csv.reader(stream, delimiter=' ')

    transcribed_dict = {}
    gold_dict = {}

    rows,columns = sobel_transform.__sobel_image__()

    for c,left,top,right,bottom,_ in box_results:
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
    with open("/home/ggdhines/gold_standard.txt","rb") as f:
        reader = csv.reader(f, delimiter=',')
        for row,column,gold_standard in reader:
            key = (int(row),int(column))

            if gold_standard == "":
                empty += 1
                if  key not in transcribed_dict:
                    correct_empty += 1
            else:
                t = sorted(transcribed_dict[key],key = lambda x:x[0])
                _,chrs = zip(*t)
                text = "".join(chrs)
                print (row,column),gold_standard,text,gold_standard==text
                if gold_standard == text:
                    total += 1
                # if gold_standard == None and text == "":
                #     correct_empty += 1
                # if gold_standard == None:
                #     print text
                #     assert Fase
                #     empty += 1

    print(total)
    print(correct_empty)
    print(empty)