from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from os import popen
import csv
import sobel_transform
__author__ = 'ggdhines'

# img = cv2.imread('/home/ggdhines/region.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def __extract_grids__(img,horizontal):
    assert len(img.shape) == 2
    # height,width = img.shape
    grid_lines = []

    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
        d_image = cv2.Sobel(img,cv2.CV_16S,0,2)

    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

        d_image = cv2.Sobel(img,cv2.CV_16S,2,0)

    d_image = cv2.convertScaleAbs(d_image)
    cv2.normalize(d_image,d_image,0,255,cv2.NORM_MINMAX)

    ret,close = cv2.threshold(d_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,th1 = cv2.threshold(d_image,127,255,cv2.THRESH_BINARY)

    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernel)

    _,contour, hier = cv2.findContours(close.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if min(h,w) > 1 and (perimeter > 500):
            s = cnt.shape
            f = np.reshape(cnt,(s[0],s[2]))

            if horizontal and (w/h > 5):
                grid_lines.append(f)

            elif not horizontal and (h/w > 5):
                grid_lines.append(f)


    return grid_lines




def __interpolate__(img,line,horizontal,background=0,foreground=255):
    if horizontal:
        x,y = zip(*line)
    else:
        y,x = zip(*line)

    y_min = min(y)
    y_max = max(y)

    # x_t,y_t= __upper_bounds__(line)


    # plt.close()
    # plt.plot(x,y,color="blue")
    #
    #
    # y_t2 = []
    # step = 60
    # for y_index in range(len(y_t)):
    #     y_t2.append(round(np.median(y_t[y_index-step:y_index+step])))
    #
    # plt.plot(x_t,y_t2,color="green")
    #
    # x_t,y_t = __lower_bounds__(line)
    #
    # y_t2 = []
    # step = 60
    # for y_index in range(len(y_t)):
    #     y_t2.append(round(np.median(y_t[y_index-step:y_index+step])))
    #
    # plt.plot(x_t,y_t2,color="green")
    #
    #
    # plt.ylim((y_max+10,y_min-10))
    # plt.xlim((0,max(x)))
    # plt.show()


    degrees = 4
    coeff = list(reversed(np.polyfit(x,y,degrees)))

    y_bar = [sum([coeff[p]*x_**p for p in range(degrees+1)]) for x_ in x]

    # plt.plot(x,y_bar)

    std = math.sqrt(np.mean([(y1-y2)**2 for (y1,y2) in zip(y,y_bar)]))

    def y_bar(x_,upper):
            return int(sum([coeff[p]*x_**p for p in range(degrees+1)]) + upper*std)

    # print f[:,0]
    domain = sorted(set(x))#sorted(set(line[:,0]))
    y_vals = [y_bar(x,-1) for x in domain]
    y_vals.extend([y_bar(x,1) for x in list(reversed(domain))])
    x_vals = list(domain)
    x_vals.extend(list(reversed(domain)))

    # plt.plot(x_vals,y_vals)
    # plt.show()

    if horizontal:
        pts = np.asarray(zip(x_vals,y_vals))
    else:
        pts = np.asarray(zip(y_vals,x_vals))

    # if (max(background,foreground) > 255) or (min(background,foreground) < 0) or (type(background) != int) or (type(foreground) != int):
    #     mask = np.zeros(img.shape)
    # else:
    mask = np.zeros(img.shape,np.uint8)
    print(background)
    mask.fill(background)
    cv2.drawContours(mask,[pts],0,255,-1)

    return mask

def __correct__(img,line,horizontal,background=0,foreground=255):
    assert len(img.shape) == 2
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,[line],0,255,-1)


    mask2 = __interpolate__(img,line,horizontal,background,foreground)

    # plt.imshow(mask)
    # plt.show(0)
    overlap = np.min([mask,mask2],axis=0)

    # plt.imshow(overlap)
    # plt.show(0)

    return mask

if __name__ == "__main__":


    horizontal_lines = __extract_grids__(gray,True)

    mask = np.zeros(gray.shape,np.uint8)
    for l in horizontal_lines:
        corrected_l = __correct__(gray,l,True)
        mask = np.max([mask,corrected_l],axis=0)

    cv2.imwrite("/home/ggdhines/testing.jpg",mask)
    vertical_lines = __extract_grids__(gray,False)
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