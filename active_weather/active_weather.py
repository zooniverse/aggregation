#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import glob
import matplotlib
# matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import database_connection
from scipy import stats
import tesseract_font
horizontal = []
import cassandra
import csv



class ActiveWeather:
    def __init__(self):
        try:
            self.cass_db = database_connection.Database()
            print("connected to the db")
        except cassandra.cluster.NoHostAvailable:
            print("could not connect to the db - will recalculate all values from scratch")
            self.cass_db = None

        # just for size reference
        # self.reference_subject = "Bear-AG-29-1940-0019"
        # self.reference_image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"+self.reference_subject+".JPG")
        # self.refer_shape = self.reference_image.shape
        #
        # self.horizontal_grid = self.cass_db.__get_horizontal_lines__(self.reference_subject,0)
        # self.vertical_grid = self.cass_db.__get_vertical_lines__(self.reference_subject,0)
        # # self.horizontal_grid,self.vertical_grid = self.__get_grid__()

        self.region = 0

        self.classifier = tesseract_font.ActiveTess()

    def __directory_to_subjects__(self,directory):
        """
        take  directory of aligned images and convert them into column based subjects for upload to Panoptes
        :param directory:
        :return:
        """
        if directory[-1] != "/":
            directory += "/"

        region_bounds = (563,3282,1276,2097)
        if self.cass_db is None:
            # we don't have a connection the db - so going to recalulate everything from scratch
            horizontal_gid,vertical_grid = self.__get_grid_for_table__(directory,region_bounds)
        else:
            # todo - read in from db
            horizontal_gid,vertical_grid = self.__get_grid_for_table__(directory,region_bounds)
            # self.horizontal_grid = self.cass_db.__get_horizontal_lines__(self.reference_subject,0)
            # self.vertical_grid = self.cass_db.__get_vertical_lines__(self.reference_subject,0)
            # todo - put this code inside the db call
            # uncomment - if you want to save the results to the cassandra db
            # self.cass_db.__add_horizontal_lines__(reference_subject,0,horizontal_lines)
            # self.cass_db.__add_vertical_lines__(reference_subject,0,vertical_lines)

        # todo - generalize to more than one region
        for fname in glob.glob(directory+"*.JPG"):
            self.__process_region__(fname,region_bounds,horizontal_gid,vertical_grid)
            break

    def __sobel_image__(self,image,horizontal):
        """
        apply the sobel operator to a given image on either the vertical or horizontal axis
        basically copied from
        http://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
        :param horizontal:
        :return:
        """
        if horizontal:
            dy = cv2.Sobel(image,cv2.CV_16S,0,2)
            dy = cv2.convertScaleAbs(dy)
            cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        else:
            dx = cv2.Sobel(image,cv2.CV_16S,2,0)
            dx = cv2.convertScaleAbs(dx)
            cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernel)

        return close

    def __contour_extraction__(self,image,horizontal):
        """
        extract all the horizontal or vertical contours from an image
        strongly inspired by
        http://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
        :param image:
        :param horizontal:
        :return:
        """
        contours_to_return = []
        _,contour, hier = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if (horizontal and w/h > 5) or ((not horizontal) and h/w > 5):
                contours_to_return.append(cnt)

        return contours_to_return

    def __get_contour_lines_over_image__(self,directory,horizontal):
        """
        return the contours lines for a subject set of already aligned subjects
        if horizontal, return only the horizontal contours. Otherwise, return the vertical contours
        returns the contours over all the image - we still have to trim to the specific region
        :param horizontal:
        :return:
        """
        # todo - currently hard coded to work with only Bear 1940
        # lined_images is the set of every aligned image after we have applied the sobel operator to it
        # i.e. extracted either the vertical or horizontal lines
        lined_images = []

        # use only the first 5 images - should be enough but we can change that if need be
        for f in glob.glob(directory+"*.JPG")[:5]:
            image = cv2.imread(f,0)
            lined_images.append(self.__sobel_image__(image,horizontal))

        # the average image is the 40th percentile
        average_image = np.percentile(lined_images,40,axis=0)
        # convert back to np.uint8 so we have a proper image
        average_image = average_image.astype(np.uint8)

        if horizontal:
            cv2.imwrite("/home/ggdhines/horizontal_average.jpg",average_image)
        else:
            cv2.imwrite("/home/ggdhines/vertical_image.jpg",average_image)

        contours_to_return = self.__contour_extraction__(average_image,horizontal)

        return contours_to_return

    def __get_grid_for_table__(self,directory,region):
        """
        directory - contains a set of aligned images
        extract the grid for a given region/table
        the region/table is specified by min_x,max_x,min_y,max_y
        :return:
        """
        assert region[0]<region[1]
        assert region[2]<region[3]
        # todo - refactor!!
        horizontal_lines = []
        vertical_lines = []
        # extract all horizontal lines
        horizontal_contours = self.__get_contour_lines_over_image__(directory,True)

        # useful for when you want to draw out the image - just for debugging

        delta = 50

        for cnt in horizontal_contours:
            shape = cnt.shape
            t = cnt.reshape((shape[0],shape[2]))
            max_x,max_y = np.max(t,axis=0)
            min_x,min_y = np.min(t,axis=0)

            if (min_y>=region[2]-delta) and (max_y<=region[3]+delta):
                # sanity check - if this an actual grid line - or just a blip?
                perimeter = cv2.arcLength(cnt,True)

                if perimeter > 100:
                    horizontal_lines.append(cnt)

        horizontal_lines.sort(key = lambda l:l[0][0][1])

        vertical_contours = self.__get_contour_lines_over_image__(directory,False)

        delta = 400
        for cnt in vertical_contours:
            shape = cnt.shape
            t = cnt.reshape((shape[0],shape[2]))
            max_x,max_y = np.max(t,axis=0)
            min_x,min_y = np.min(t,axis=0)

            interior_line = (min_x >= region[0]-100) and (max_x <= region[1]+100)and(min_y>=region[2]-delta) and (max_y<=region[3]+delta)
            through_line = (min_x >= region[0]-100) and (max_x <= region[1]+100) and (min_y < region[2]) and(max_y > region[3])

            if interior_line or through_line:

                perimeter = cv2.arcLength(cnt,True)
                if perimeter > 1000:
                    # cv2.drawContours(masks,[cnt],0,255,3)
                    vertical_lines.append(cnt)


        vertical_lines.sort(key = lambda l:l[0][0][0])

        return horizontal_lines,vertical_lines

    def __extract_column__(self,column_index):
        image = cv2.imread("/home/ggdhines/gaussian.jpg",0)

        # get the region coordinates - so we can convert global grid line coordinates to
        # local ones (relative to just the grid line)
        _,_,region_x,_,_ = self.__region_mask__()
        t = self.vertical_grid[column_index]
        min_x,_ = np.min(t,axis=0)
        t = self.vertical_grid[column_index+1]
        max_x,_ = np.max(t,axis=0)

        # print(image[:,5:7])
        column = image[:,(min_x-region_x):(max_x-region_x+1)]

        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # remove any "blips"
        _,contour, hier = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)

            print((h,w))
            if (h <5) or (w < 5):
                cv2.drawContours(column,[cnt],0,255,-1)

        cv2.imwrite("/home/ggdhines/column"+str(column_index)+".jpg",column)

    # def __process_row__(self,row_index):
    #     min_y,max_y,min_x,max_x ,mask = self.__region_mask__()
    #     image = cv2.imread("/home/ggdhines/gaussian.jpg",0)
    #     print(image.shape)
    #     # most_common_pigment = int(stats.median(image,axis=None)[0][0])
    #
    #     region_y,_,_,_,_ = self.__region_mask__()
    #
    #     t = self.horizontal_grid[row_index]#.reshape((shape[0],shape[2]))
    #     _,min_y = np.min(t,axis=0)
    #
    #     t = self.horizontal_grid[row_index+1]#.reshape((shape[0],shape[2]))
    #     _,max_y = np.max(t,axis=0)
    #
    #     row = image[(min_y-region_y):(max_y-region_y+1),:]
    #
    #     # wedge masks
    #     big_mask = np.zeros(row.shape,np.uint8)
    #     big_mask.fill(255)
    #
    #     t = self.horizontal_grid[row_index]#.reshape((shape[0],shape[2]))
    #     _,min_y = np.max(t,axis=0)
    #
    #     t = self.horizontal_grid[row_index+1]#.reshape((shape[0],shape[2]))
    #     _,max_y = np.min(t,axis=0)
    #
    #
    #
    #     # mask3[min_y:max_y+1,min_x:max_x+1] = 255
    #     # mask4 = cv2.bitwise_xor(mask3,mask)
    #
    #     cv2.imwrite("/home/ggdhines/row"+str(row_index)+".jpg",row)
    #
    #
    #
    #     # row_contents = cv2.bitwise_and(image,mask1)
    #     # background = np.where(mask2>0)
    #     # row_contents[background] = most_common_pigment
    #     #
    #     # # remove the vertical rows
    #     # for v in self.vertical_grid:
    #     #     cv2.drawContours(row_contents,[v],0,255,-1)
    #     #
    #     # row = row_contents[min_y:max_y+1,min_x:max_x+1]
    #     #
    #     # ret2,th2 = cv2.threshold(row,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     #
    #     # cv2.imwrite("/home/ggdhines/row.jpg",th2)
    #     #
    #     #
    #     # # # row_ = cv2.adaptiveThreshold(row,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,2)
    #     # # ret,row_ = cv2.threshold(row,most_common_pigment+5,255,cv2.THRESH_BINARY)
    #     # # cv2.imwrite("/home/ggdhines/row_.jpg",row_)
    #     #
    #     # _,contour, hier = cv2.findContours(th2.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #     # for cnt,h_ in zip(contour,hier[0]):
    #     #
    #     #     x,y,w,h = cv2.boundingRect(cnt)
    #     #
    #     #     # print(h)
    #     #     # print(w/h)
    #     #     # print("")
    #     #     if (h < 10):
    #     #         print(h)
    #     #         cv2.drawContours(th2,[cnt],0,255,-1)
    #     #
    #     #
    #     # cv2.imwrite("/home/ggdhines/row2.jpg",th2)
    #     #
    #     #
    #     # row_colour = np.zeros((th2.shape[0],th2.shape[1],3),np.uint8)
    #     # row_colour[:,:,0] = th2
    #     # row_colour[:,:,1] = th2
    #     # row_colour[:,:,2] = th2
    #     #
    #     # tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/",language="eng")
    #     # tess.tessedit_char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890."
    #     # tess.set_image(row_colour)
    #     # tess.get_utf8_text()
    #     # words = list(tess.words())
    #     # words_in_cell = [w.text for w in words if w.text is not None]
    #     # print(words_in_cell)
    #     # conf_in_cell = [w.confidence for w in words if w.text is not None]
    #     # print(conf_in_cell)
    #     #
    #     # # words,confidence = hocr.scan()
    #     # # for w,c in zip(words,confidence):
    #     # #     print((w,c))
    #     # raw_input("check row.jpg")

    def __region_mask__(self,reference_image,horizontal_grid,vertical_grid):
        """
        use the first and last horizontal/vertical grid lines to make a mask around the desired region/table
        :return:
        """
        reference_shape = reference_image.shape
        # [:2] in case we read in the image in colour format - doesn't seem necessary to throw an error
        # the first mask will be an outline of the region, sort of like #. The second mask will fill in the
        # central interior box
        mask = np.zeros(reference_shape[:2],np.uint8)
        mask2 = np.zeros(mask.shape,np.uint8)
        # draw the first and last horizontal/vertical grid lines to create a box
        cv2.drawContours(mask,horizontal_grid,0,255,-1)
        cv2.drawContours(mask,horizontal_grid,len(horizontal_grid)-2,255,-1)
        cv2.drawContours(mask,vertical_grid,0,255,-1)
        cv2.drawContours(mask,vertical_grid,len(vertical_grid)-1,255,-1)

        # find the (hopefully) one interior contour - should be our mask
        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        assert len(contours) == 1
        for c,h in zip(contours,hier[0]):
            if h[-1] == -1:
                continue

            cv2.drawContours(mask2,[c],0,255,-1)

        return mask2

    # def __remove_grid__(self,image,replacement_colour):
    #     """
    #     replace all of the grid with either the desired colour - probably either white
    #     of the most common colour (assuming that corresponds to background colour)
    #     :param image:
    #     :param replacement_colour:
    #     :return:
    #     """
    #     cv2.drawContours(image,self.horizontal_grid,-1,replacement_colour,-1)
    #     cv2.drawContours(image,self.vertical_grid,-1,replacement_colour,-1)

    def __process_region__(self,fname,region_bounds,horizontal_grid,vertical_grid):
        """
        main function - open fname, "zoom in" on the desired region, apply thresholding to "clean it up"
        and then split by column
        region_bounds = min_x,max_x,min_y,max_y
        :param fname:
        :param region:
        :param mask:
        :return:
        """
        image = cv2.imread(fname,0)
        # blank out anything that is not inside the given mask
        # masked_image = cv2.bitwise_and(image,mask)
        # "zoom " in
        # region = masked_image[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[3]+1]

        # uncomment if you want to apply ostu thresholding
        # see http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0
        # ret2,thresh1 = cv2.threshold(region,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # apply Gaussian filtering - you can play around with 201 - needs to be odd and probably big enough
        # to cover a full cell
        gaussian_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,2)
        # remove all grid lines by "whiting" them out
        cv2.drawContours(gaussian_image,horizontal_grid,-1,255,-1)
        cv2.drawContours(gaussian_image,vertical_grid,-1,255,-1)
        # "zoom in"
        region = gaussian_image[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]

        cv2.imwrite("/home/ggdhines/example.jpg",region)

        image = self.__image_clean__(region)

        cv2.imwrite("/home/ggdhines/cleaned_example.jpg",image)


    def __image_clean__(self,image):
        """
        after removing grid lines and applying thresholding, we will probably still have small "ticks" - bits of the
        grid line which weren't removed but can still cause problems for Tesseract (and probably other approaches too)
        """
        _,contours, hier = cv2.findContours(image.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            print(h)
            if (h <= 10) or (perimeter <= 50):
                cv2.drawContours(image,[cnt],0,255,-1)

        return image

    def __read_box__(self):
        image = cv2.imread("/home/ggdhines/step4.jpg")
        s = image.shape
        with open("/home/ggdhines/boxout","r") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            for row in spamreader:
                _,x1,y1,x2,y2,_ = row
                cv2.rectangle(image,(int(x1),s[0]-int(y1)),(int(x2),s[0]-int(y2)),(255,0,0),2)

        image = cv2.imwrite("/home/ggdhines/step5.jpg",image)


    # def __row_mask__(self,row_index):
    #     """
    #     creates two masks - for one extracting the cell contents
    #     the other for allowing a proper rectangle to be formed - the second mask will be used
    #     for setting the background colour
    #     :param fname:
    #     :param row_index:
    #     :return:
    #     """
    #     # image = cv2.imread(fname,0)
    #     # most_common_pigment = int(stats.mode(image,axis=None)[0][0])
    #     # image = cv2.drawContours(image,self.horizontal_grid,-1,most_common_pigment,-1)
    #     # image = cv2.drawContours(image,self.vertical_grid,-1,most_common_pigment,-1)
    #
    #
    #     mask = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
    #     cv2.drawContours(mask,self.horizontal_grid,row_index,255,-1)
    #     cv2.drawContours(mask,self.horizontal_grid,row_index+1,255,-1)
    #     cv2.drawContours(mask,self.vertical_grid,0,255,-1)
    #     cv2.drawContours(mask,self.vertical_grid,len(self.vertical_grid)-1,255,-1)
    #     cv2.imwrite("/home/ggdhines/mask.jpg",mask)
    #
    #     # _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    #
    #     # contours are probably in sorted order but just to be sure
    #     mask2 = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
    #     for c,h in zip(contours,hier[0]):
    #         if h[-1] == -1:
    #             continue
    #
    #         cv2.drawContours(mask2,[c],0,255,-1)
    #
    #     t = self.horizontal_grid[row_index]#.reshape((shape[0],shape[2]))
    #     _,min_y = np.min(t,axis=0)
    #
    #     t = self.horizontal_grid[row_index+1]#.reshape((shape[0],shape[2]))
    #     _,max_y = np.max(t,axis=0)
    #
    #     # repeat for vertical grid lines
    #     # shape = self.vertical_grid[v_index].shape
    #     t = self.vertical_grid[0]#.reshape((shape[0],shape[2]))
    #     min_x,_ = np.max(t,axis=0)
    #
    #     t = self.vertical_grid[-1]#.reshape((shape[0],shape[2]))
    #     max_x,_ = np.min(t,axis=0)
    #
    #     mask3 = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
    #     mask3[min_y:max_y+1,min_x:max_x+1] = 255
    #     cv2.imwrite("/home/ggdhines/mask3.jpg",mask3)
    #
    #     mask4 = cv2.bitwise_xor(mask3,mask2)
    #     cv2.imwrite("/home/ggdhines/mask4.jpg",mask4)
    #
    #     return min_x,max_x,min_y,max_y,mask2,mask4
    #     # assert False
    #     #
    #     #
    #     # # print((min_x,min_y,max_x,max_y))
    #     #
    #     # row = image[min_y:max_y,min_x:max_x]
    #     #
    #     # # colour_image = np.zeros((row.shape[0],row.shape[1],3),np.uint8)
    #     # # colour_image[:,:,0] = row
    #     # # colour_image[:,:,1] = row
    #     # # colour_image[:,:,2] = row
    #     #
    #     # cv2.imwrite("/home/ggdhines/output.jpg",row)
    #     #
    #     # words,confidence = hocr.scan()
    #     # print(words)

    # def __extract_cell__(self,h_index,v_index,colour=True):
    #     """
    #     :param image: can use an approximate image which is based on a threshold using more global values
    #     better for determining whether or a cell is empty. If we know that a cell is not empty, we can do thresholding
    #     based on more local values
    #     :param h_index:
    #     :param v_index:
    #     :return:
    #     """
    #     # todo - DEFINITELY precalculate these
    #     # shape = self.horizontal_grid[h_index].shape
    #     t = self.horizontal_grid[h_index]#.reshape((shape[0],shape[2]))
    #     _,min_y = np.max(t,axis=0)
    #
    #     # shape = self.horizontal_grid[h_index+1].shape
    #     t = self.horizontal_grid[h_index+1]#.reshape((shape[0],shape[2]))
    #     _,max_y = np.min(t,axis=0)
    #
    #     # repeat for vertical grid lines
    #     # shape = self.vertical_grid[v_index].shape
    #     t = self.vertical_grid[v_index]#.reshape((shape[0],shape[2]))
    #     min_x,_ = np.max(t,axis=0)
    #
    #     # shape = self.vertical_grid[v_index+1].shape
    #     t = self.vertical_grid[v_index+1]#.reshape((shape[0],shape[2]))
    #     max_x,_ = np.min(t,axis=0)
    #
    #     # mask = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
    #     # cv2.drawContours(mask,self.horizontal_grid,h_index,255,-1)
    #     # cv2.drawContours(mask,self.horizontal_grid,h_index+1,255,-1)
    #     # cv2.drawContours(mask,self.vertical_grid,v_index,255,-1)
    #     # cv2.drawContours(mask,self.vertical_grid,v_index+1,255,-1)
    #     #
    #     # _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     #
    #     # # contours are probably in sorted order but just to be sure
    #     # for c,h in zip(contours,hier[0]):
    #     #     if h[-1] != -1:
    #     #         continue
    #     #
    #     #     cv2.drawContours(mask,[c],0,255,-1)
    #     #
    #     # res = cv2.bitwise_and(mask,image)
    #     #
    #     # # now go back over and draw in contours in white - any black inside the cell that is on a grid
    #     # # line is mostly likely grid, not ink
    #     # # most of the time this wouldn't make a difference - but be sure
    #     # # todo - double check
    #     # temp_res = res[min_y:max_y+1,min_x:max_x+1]
    #     # most_common_pigment = int(stats.mode(temp_res,axis=None)[0][0])
    #     # # print most_common_pigment
    #     # # print type(most_common_pigment)
    #     #
    #     # cv2.drawContours(res,self.horizontal_grid,h_index,most_common_pigment,-1)
    #     # cv2.drawContours(res,self.horizontal_grid,h_index+1,most_common_pigment,-1)
    #     # cv2.drawContours(res,self.vertical_grid,v_index,most_common_pigment,-1)
    #     # cv2.drawContours(res,self.vertical_grid,v_index+1,most_common_pigment,-1)
    #
    #     # now that we have cleaned up this cell, actually zoom into it
    #     image = cv2.imread("/home/ggdhines/gaussian.jpg",0)
    #     min_y -= 1276
    #     max_y  -= 1276
    #     min_x -= 572
    #     max_x -= 572
    #     # print(image)
    #     # print(image.shape)
    #     cell = image[min_y:max_y+1,min_x:max_x+1]
    #     cell = image[min_y:max_y+1,:]
    #     # print((min_y,max_y,min_x,max_x))
    #     # print(cell)
    #     cv2.imwrite("/home/ggdhines/cell_"+str(h_index)+ "_" + str(v_index) + ".jpg",cell)
    #     # plt.imshow(cell)
    #     # plt.show()
    #
    #     # # tesseract needs things in colour, so convert
    #     #
    #     # boundary = [min_y,max_y+1,min_x,max_x+1]
    #     #
    #     # if colour:
    #     #     colour_res = np.zeros((res.shape[0],res.shape[1],3),np.uint8)
    #     #     colour_res[:,:,0] = res[:,:]
    #     #     colour_res[:,:,1] = res[:,:]
    #     #     colour_res[:,:,2] = res[:,:]
    #     #
    #     #     return colour_res,boundary
    #     # else:
    #     #     return res,boundary

    def __extract_table__(self,fname):
        """
        extract and save the whole table (for a given region) from an image
        if a box file exists, plot the bounding boxes
        :param fname:
        :return:
        """
        t = self.horizontal_grid[0]#.reshape((shape[0],shape[2]))
        _,min_y = np.max(t,axis=0)

        # shape = self.horizontal_grid[h_index+1].shape
        t = self.horizontal_grid[-1]#.reshape((shape[0],shape[2]))
        _,max_y = np.min(t,axis=0)

        # repeat for vertical grid lines
        # shape = self.vertical_grid[v_index].shape
        t = self.vertical_grid[0]#.reshape((shape[0],shape[2]))
        min_x,_ = np.max(t,axis=0)

        # shape = self.vertical_grid[v_index+1].shape
        t = self.vertical_grid[-1]#.reshape((shape[0],shape[2]))
        max_x,_ = np.min(t,axis=0)

        image = cv2.imread(fname)

        region = image[min_y:max_y+1,min_x:max_x+1]
        cv2.imwrite("/home/ggdhines/region.jpg",region)
        s = region.shape

        with open("/home/ggdhines/region.box","r") as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            for row in reader:
                _,x1,y1,x2,y2,_ = row
                print(((int(x1),s[0]-int(y1)),(int(x2),s[0]-int(y2))))
                region = cv2.rectangle(region,(int(x1),s[0]-int(y1)),(int(x2),s[0]-int(y2)),(255,0,0),2)

        cv2.imwrite("/home/ggdhines/region_with_boxes.jpg",region)

    # def __image_threshold__(self,fname):
    #     image = cv2.imread(fname,0)
    #     gaussian_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,2)
    #
    #     cv2.imwrite("/home/ggdhines/gaussian.jpg",gaussian_image)

    # def __process_image__(self,fname):
    #     # todo - generalize to more than one region
    #     for region_id in range(1):
    #         self.__process_region__(fname,)
        # self.__region_mask__()
        # subject_id = fname.split("/")[-1][:-4]
        #
        # image = cv2.imread(fname,0)
        #
        # approximate_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,0)

        # for h_index in range(len(self.horizontal_grid)-1):

            # self.__process_row__(fname,h_index)


            # for v_index in range(len(self.vertical_grid)-1):
            #
            #     # start by just checking if that cell is empty or not
            #     cell,[lb_y,ub_y,lb_x,ub_x] = self.__extract_cell__(image,h_index,v_index)
            #
            #     context = 0
            #     apprximate_cell = approximate_image[lb_y-context:ub_y+context,lb_x-context:ub_x+context]
            #
            #     self.classifier.tess.set_image(cell)
            #     self.classifier.tess.get_utf8_text()
            #
            #     cell,[lb_y,ub_y,lb_x,ub_x] = self.__extract_cell__(image,h_index,v_index)
            #     cv2.imwrite("/home/ggdhines/output.jpg",cell)
            #     # t = pytesseract.image_to_string(Image.open('/home/ggdhines/output.jpg'))
            #     # print("t is " + t)
            #
            #     # sanity_string = pytesseract.image_to_string(Image.fromarray(cell))
            #
            #     # print(type(Image.open('/home/ggdhines/output.png')))
            #
            #     words = list(self.classifier.tess.words())
            #     # for line in self.classifier.tess.text_lines():
            #     #     print(line.text)
            #     os.system("tesseract /home/ggdhines/output.jpg /tmp/sanity_check > /dev/null 2> /dev/null")
            #     l = ""
            #     with open("/tmp/sanity_check.txt") as f:
            #         for i in f.readlines():
            #             l += i
            #
            #     print(l.strip("\n"))
            #
            #     words_in_cell = [w.text for w in words if w.text is not None]
            #     word = "".join(words_in_cell)
            #     print("word is " + word)
            #     raw_input("whatever")
            #     continue
            #     conf_in_cell = [w.confidence for w in words if w.text is not None]
            #
            #     # todo - can we get more detail knowing that the cell is not empty?
            #     # # is there something in the cell, if so extract that cell in full detail
            #     #
            #     # if words != [None]:
            #     #     cell = self.__extract_cell__(image,h_index,v_index,colour=False)
            #     #     _,cell = cv2.threshold(cell,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #
            #     # tesseract might split up the word into multiple parts - usually a sign that something has
            #     # gone wrong but we'll see
            #
            #     print(str(sanity_string) == str(word))
            #     print((word,sanity_string,conf_in_cell))
            #
            #     if self.cass_db.__has_cell_been_transcribed__(subject_id,self.region,h_index,v_index):
            #         print("skipping")
            #         continue
            #
            #     if word != "":
            #         context = 50
            #         cell_with_context = image[lb_y-context:ub_y+context,lb_x-context:ub_x+context]
            #         cv2.imwrite("/home/ggdhines/cell.jpg",cell_with_context)
            #         overall_confidence = np.mean(conf_in_cell)
            #         cell_string = raw_input("Look at cell.jpg - enter its contents: ")
            #         if cell_string in ["","\n"]:
            #             continue
            #
            #         # convert to grayscale - the RGB values are all the same
            #         digits,minimum_y = extract_digits.extract(cell[:,:,0])
            #
            #         cell_list = [c for c in cell_string]
            #
            #         self.cass_db.__add_gold_standard__(subject_id,self.region,h_index,v_index,cell_string)
            #         if len(cell_list) == len(digits):
            #             print("adding learning cases")
            #             for index in range(len(cell_list)):
            #                 top_of_row = min(minimum_y)
            #                 offset = minimum_y[index] - top_of_row
            #                 self.cass_db.__add_character__(subject_id,self.region,h_index,v_index,index,cell_string[index],digits[index],offset)
            #             # print "done adding"
            #
            #             # classifier.__add_characters__(cell_list,digits,minimum_y)
            #             # classifier.__update_tesseract__()

if __name__ == "__main__":
    project = ActiveWeather()
    project.__directory_to_subjects__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/")
    # project.__image_threshold__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__extract_table__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__extract_column__(8)
    # project.__process_image__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # for i in range(10):
    #     project.__process_row__(i)
    # project.__process_image__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__remove_template__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")\
    # project.__read_box__()