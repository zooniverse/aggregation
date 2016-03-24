import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from os import popen
from active_weather import ActiveWeather
import numpy as np

directory = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"

# min_x,max_x,min_y,max_y
region_bounds = (559,3282,1276,2097)
project = ActiveWeather()
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



image = cv2.imread("/home/ggdhines/region.jpg",0)
# print img
# radius = 15
# selem = disk(radius)
#
# local_otsu = rank.otsu(img, selem)
# threshold_global_otsu = threshold_otsu(img)
# global_otsu = img >= threshold_global_otsu
#
#
# fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
# ax1, ax2, ax3, ax4 = ax.ravel()
#
# fig.colorbar(ax1.imshow(img, cmap=plt.cm.gray),
#              ax=ax1, orientation='horizontal')
# ax1.set_title('Original')
# ax1.axis('off')
#
# fig.colorbar(ax2.imshow(local_otsu, cmap=plt.cm.gray),
#              ax=ax2, orientation='horizontal')
# ax2.set_title('Local Otsu (radius=%d)' % radius)
# ax2.axis('off')
#
# ax3.imshow(img >= local_otsu, cmap=plt.cm.gray)
# ax3.set_title('Original >= Local Otsu' % threshold_global_otsu)
# ax3.axis('off')
#
# ax4.imshow(global_otsu, cmap=plt.cm.gray)
# ax4.set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
# ax4.axis('off')
#
# plt.show()

ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(image,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(th2)
plt.show()

plt.imshow(th3)
plt.show()

cv2.imwrite("/home/ggdhines/testing3.jpg",th2)
cv2.imwrite("/home/ggdhines/testing4.jpg",th3)

stream = popen("tesseract -psm 6 /home/ggdhines/testing4.jpg stdout makebox")
contents = stream.readlines()


transcribed = []
height,width = image.shape

for row in contents:
    try:
        c,left,top,right,bottom,_ = row[:-1].split(" ")
    except ValueError:
        print(row)
        raise

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
    cv2.polylines(image,[l],True,(0,255,0))

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
cv2.imwrite("/home/ggdhines/test.jpg",image)

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
total = 0

for k in transcribed_dict:
    text_with_coords = zip(transcribed_dict[k][0],transcribed_dict[k][1])
    text_with_coords.sort(key = lambda x:x[0])
    _,text_list = zip(*text_with_coords)
    text = "".join(text_list)

    print(text,gold_dict[k],text==gold_dict[k])
    if text==gold_dict[k]:
        total += 1

print(total)