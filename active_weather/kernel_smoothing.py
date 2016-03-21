import numpy as np
import cv2
import csv
from os import popen
import paper_quad
import sobel_transform
import math
import tesserpy
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
        x,y = zip(*line)
        line = zip(y,x)
        x,y = __upper_bounds__(line)
        x2,y2 = __lower_bounds__(line)

    x2 = list(reversed(x2))
    y2 = list(reversed(y2))


    x_pts = x
    x_pts.extend(x2)

    y_pts = []
    step = 150
    for y_index in range(len(y)):
        m = np.median(y[max(y_index-step,0):y_index+step])
        assert not math.isnan(m)

        y_pts.append(int(round(m)))

    for y_index in range(len(y2)):
        m = np.median(y2[max(y_index-step,0):y_index+step])
        assert not math.isnan(m)

        y_pts.append(int(round(m)))

    mask = np.zeros(img.shape,np.uint8)

    # print zip(x,y)

    if horizontal:
        pts = np.asarray(zip(x_pts,y_pts))
    else:
        pts = np.asarray(zip(y_pts,x_pts))

    cv2.drawContours(mask,[pts],0,255,-1)
    # cv2.imshow("img",mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # assert False

    return mask

def __correct__(img,line,horizontal,background=0,foreground=255):
    assert len(img.shape) == 2
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask,[line],0,255,-1)


    mask2 = __median_kernel__(img,line,horizontal)

    # plt.imshow(mask)
    # plt.show(0)

    overlap = np.min([mask,mask2],axis=0)

    # plt.imshow(overlap)
    # plt.show(0)

    return overlap


def __mask_lines__(gray):
    # img = cv2.imread('/home/ggdhines/region.jpg')
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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

    # cv2.imshow("img",mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret,thresh1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
    masked_image = np.max([thresh1,mask],axis=0)

    return masked_image

def __gen_columns__(masked_image,gray):
    vertical_contours = paper_quad.__extract_grids__(gray,False)

    vertical_lines = []

    for v in vertical_contours:
        x_val,_ = np.mean(v,axis=0)
        vertical_lines.append(x_val)

    vertical_lines.extend([0,img.shape[1]])
    vertical_lines.sort()
    for column_index in range(len(vertical_lines)-1):
        sub_image = masked_image[:,vertical_lines[column_index]:vertical_lines[column_index+1]]

        colour_sub_image = np.zeros([sub_image.shape[0],sub_image.shape[1],3],np.uint8)
        colour_sub_image[:,:,0] = sub_image
        colour_sub_image[:,:,1] = sub_image
        colour_sub_image[:,:,2] = sub_image
        yield colour_sub_image

def __ocr_image__(image):
    tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/",language="eng")
    # print(vars(tess))
    tess.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_BLOCK
    # tess.tessedit_ocr_engine_mode = tesserpy.OEM_CUBE_ONLY
    # tess.tessedit_page_iteratorlevel = tess.RIL_SYMBOL
    tess.tessedit_char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.abcdefghijkmnopqrstuvwxyz"

    tess.set_image(image)
    tess.get_utf8_text()
    transcribed = []

    # image_height,image_width = image.shape[:2]

    for word in tess.symbols():
        # print(word.text,word.confidence)
        bb = word.bounding_box
        height = abs(bb.top-bb.bottom)
        width = bb.right - bb.left

        if min(height,width) >= 10:

            transcribed.append((word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
            # print("{}\t{}\tt:{}; l:{}; r:{}; b:{}".format(word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
        # confidences.append(word.confidence)
        # text.append(word.text)
        # boxes.append(word.bounding_box)
    return transcribed

def __cell_boundaries__(image):
    horizontal_grid = []
    vertical_grid = []

    height,width = image.shape[:2]

    horizontal_lines = paper_quad.__extract_grids__(image,True)

    # plt.imshow(image)
    for row_index in range(len(horizontal_lines)):
        c = np.median(horizontal_lines[row_index],axis=0)[1]
        # lb = np.max(horizontal_lines[row_index+1],axis=0)[1]

        # if horizontal_grid
        horizontal_grid.append(c)

        # plt.plot((0,width),(lb,lb))
        # plt.plot((0,width),(ub,ub))

    horizontal_grid.extend([0,height])
    horizontal_grid.sort()

    vertical_lines = paper_quad.__extract_grids__(image,False)
    for column_index in range(len(vertical_lines)):
        c = np.median(vertical_lines[column_index],axis=0)[0]
        # ub = np.max(vertical_lines[column_index+1],axis=0)[0]

        vertical_grid.append(c)

        # plt.plot((lb,lb),(0,height))
        # plt.plot((ub,ub),(0,height))
    vertical_grid.extend([0,width])
    vertical_grid.sort()


    return horizontal_grid,vertical_grid

def __place_in_cell__(transcriptions,horizontal_grid,vertical_grid,image):
    cell_contents = {}

    for (t,c,top,left,right,bottom) in transcriptions:
        print(t,c)
        mid_y = (int(top)+int(bottom))/2.
        mid_x = (int(right)+int(left))/2.

        in_row = False

        for row_index in range(len(horizontal_grid)-1):
            lb = horizontal_grid[row_index]
            ub = horizontal_grid[row_index+1]
            if lb <= mid_y <= ub:
                in_row = True
                break

        assert in_row

        in_column = False
        for column_index in range(len(vertical_grid)-1):
            lb = vertical_grid[column_index]
            ub = vertical_grid[column_index+1]
            if lb <= mid_x <= ub:
                in_column = True
                break
        assert in_column

        # plt.imshow(image,cmap="gray")
        # plt.plot(mid_x,mid_y,"o")
        # plt.show()

        key = (column_index,row_index)
        if key not in cell_contents:
            cell_contents[key] = [(mid_x,t,c)]
        else:
            cell_contents[key].append((mid_x,t,c))

    for key in cell_contents:
        sorted_contents = sorted(cell_contents[key], key = lambda x:x[0])
        _,text,confidence = zip(*sorted_contents)
        text = "".join(text)
        confidence = min(confidence)
        cell_contents[key] = (text,confidence)

    print(cell_contents)

if __name__ == "__main__":
    img = cv2.imread('/home/ggdhines/region.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    horizontal_grid,vertical_grid = __cell_boundaries__(gray)
    # assert False

    masked_image = __mask_lines__(gray)

    transcriptions = __ocr_image__(masked_image)
    __place_in_cell__(transcriptions,horizontal_grid,vertical_grid,gray)

    # for column_image in __gen_columns__(masked_image,gray):
    #     __ocr_image__(column_image)
    #     print

    assert False



    cv2.imwrite("/home/ggdhines/masked.jpg",masked_image)

    #######
    ######

    stream = popen("tesseract -psm 6 /home/ggdhines/masked.jpg stdout makebox")
    # box_results = csv.reader(stream, delimiter=' ')

    transcribed = []

    height,width = gray.shape

    # print(box_results[0])

    # for l in stream.readlines():
    #     parsed_l = l[:-1].split(" ")
    #     c,left,top,right,bottom,_ = parsed_l
    #
    #     top = height - int(top)
    #     bottom = height - int(bottom)
    #     # print(top,bottom)
    #     assert top > 0
    #     assert bottom > 0
    #     left = int(left)
    #     right = int(right)
    #
    #
    #     if min((top-bottom),(right-left)) <= 6:
    #         l_y = [top,top,bottom,bottom,top]
    #         l_x = [left,right,right,left,left]
    #         l = np.asarray(zip(l_x,l_y))
    #         # print(l)
    #         cv2.drawContours(masked_image,[l],0,255,-1)

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