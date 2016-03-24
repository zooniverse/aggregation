import csv
from os import popen
import sobel_transform
import cv2

img = cv2.imread("/home/ggdhines/sobel_masked.jpg")
height,width,_ = img.shape

stream = popen("tesseract -psm 6 /home/ggdhines/sobel_masked.jpg stdout makebox")
box_results = csv.reader(stream, delimiter=' ')

transcribed_dict = {}
gold_dict = {}

rows,columns = sobel_transform.__sobel_image__()

for c,left,top,right,bottom,_ in box_results:
    print c,left,top,right,bottom
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

print total
print correct_empty
print empty