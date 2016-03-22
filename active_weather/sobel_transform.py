# import matplotlib
# matplotlib.use('WXAgg')
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import popen
import csv
# from active_weather import ActiveWeather

# directory = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"

# min_x,max_x,min_y,max_y
# region_bounds = (559,3282,1276,2097)
# project = ActiveWeather()

# project.cass_db.__get_subjects__()

# fname = directory+"Bear-AG-29-1940-0720.JPG"



# horizontal_grid,vertical_grid = project.__get_grid_for_table__(directory,region_bounds,fname)
#
# rows = []
# columns = []
#
# for row_index in range(len(horizontal_grid)-1):
#     lb = np.min(horizontal_grid[row_index],axis=0)[1]-region_bounds[2]
#     ub = np.max(horizontal_grid[row_index+1],axis=0)[1]-region_bounds[2]
#
#     rows.append((lb,ub))
# print(len(rows))
#
# for column_index in range(len(vertical_grid)-1):
#     lb = np.min(vertical_grid[column_index],axis=0)[0]-region_bounds[0]
#     ub = np.max(vertical_grid[column_index+1],axis=0)[0]-region_bounds[0]
#
#     columns.append((lb,ub))

def __sobel_image__():
    img = cv2.imread('/home/ggdhines/region.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height,width,_ = img.shape
    ret,thresh1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)

    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    dy = cv2.Sobel(gray,cv2.CV_16S,0,2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

    cv2.imwrite("/home/ggdhines/horizontal.jpg",close)

    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    dy = cv2.Sobel(gray,cv2.CV_16S,0,2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

    cv2.imwrite("/home/ggdhines/horizontal.jpg",close)

    mask = np.zeros(close.shape,np.uint8)
    mask.fill(0)
    rows = []
    _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if (w/h > 5) and min(h,w) > 1 and (perimeter > 500):
            s = cnt.shape
            f = np.reshape(cnt,(s[0],s[2]))
            _,row = np.mean(f,axis=0)
            rows.append(row)
            cv2.drawContours(mask,[cnt],0,255,-1)

    # plt.imshow(mask,cmap="gray")
    # plt.show()
    rows.sort()
    rows.insert(0,0)
    rows.append(height)

    cv2.imwrite("/home/ggdhines/horiz_sobel.jpg",mask)

    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

    dx = cv2.Sobel(gray,cv2.CV_16S,1,0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    _,contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    columns = []
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if (h/w > 5) and min(h,w)>1 and (perimeter > 300):
            s = cnt.shape
            f = np.reshape(cnt,(s[0],s[2]))
            column,_ = np.mean(f,axis=0)
            columns.append(column)
            cv2.drawContours(mask,[cnt],0,255,-1)
    columns.sort()
    columns.insert(0,0)
    columns.append(width)
    masked_image = np.max([thresh1,mask],axis=0)

    # plt.imshow(masked_image,cmap="gray")
    # plt.show()
    cv2.imwrite("/home/ggdhines/masked.jpg",masked_image)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
    closey = close.copy()

    stream = popen("tesseract -psm 6 /home/ggdhines/masked.jpg stdout makebox")
    box_results = csv.reader(stream, delimiter=' ')

    # backtorgb = cv2.cvtColor(masked_image.shape,cv2.COLOR_GRAY2RGB)
    blank = np.zeros(masked_image.shape,np.uint8)
    for c,left,top,right,bottom,_ in box_results:
        top = height - int(top)
        bottom = height - int(bottom)



        assert top > 0
        assert bottom > 0
        left = int(left)
        right = int(right)

        h = top-bottom
        w = right-left
        if min(h,w) <= 10:

            l_y = [top,top,bottom,bottom,top]
            l_x = [left,right,right,left,left]
            l = np.asarray(zip(l_x,l_y))
            # cv2.polylines(backtorgb,[l],True,(0,255,0))
            cv2.drawContours(blank,[l],0,255,2)
            cv2.drawContours(masked_image,[l],0,255,-1)
        else:
            pass

    cv2.imwrite("/home/ggdhines/sobel_masked.jpg",masked_image)
    cv2.imwrite("/home/ggdhines/blank.jpg",blank)
    return rows,columns

if __name__ == "__main__":
    __sobel_image__()