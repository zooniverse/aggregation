#!/usr/bin/env python
import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt

import extract_digits
from scipy import stats
import tesseract_font

# tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/", language="eng")

horizontal = []

# just for size reference
reference_image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0019.JPG")
refer_shape = reference_image.shape

classifier = tesseract_font.ActiveTess()


class ActiveWeather:
    def __init__(self):
        self.horizontal_grid,self.vertical_grid = self.__get_grid__()

    def __get_lines__(self,horizontal):
        # todo - swap
        lined_images = []

        for f in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:5]:
            img = cv2.imread(f,0)

            if horizontal:
                dy = cv2.Sobel(img,cv2.CV_16S,0,2)
                dy = cv2.convertScaleAbs(dy)
                cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
                ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
            else:
                dx = cv2.Sobel(img,cv2.CV_16S,2,0)
                dx = cv2.convertScaleAbs(dx)
                cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
                ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

            close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernel)

            lined_images.append(close)

        average_image = np.percentile(lined_images,40,axis=0)

        average_image = average_image.astype(np.uint8)

        cv2.imwrite("/home/ggdhines/results.jpg",average_image)

        contours_to_return = []

        _,contour, hier = cv2.findContours(average_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if (horizontal and w/h > 5) or ((not horizontal) and h/w > 5):
                cv2.drawContours(average_image,[cnt],0,255,-1)
                contours_to_return.append(cnt)
            else:
                cv2.drawContours(average_image,[cnt],0,0,-1)

        average_image = cv2.morphologyEx(average_image,cv2.MORPH_DILATE,None,iterations = 2)

        return average_image,contours_to_return


    def __get_grid__(self):
        # extract all horizontal lines
        _,horizontal_contours = self.__get_lines__(True)

        horizontal_lines = []
        vertical_lines = []

        # useful for when you want to draw out the image - just for debugging
        template_image = np.zeros(refer_shape,np.uint8)
        delta = 50

        for cnt in horizontal_contours:
            shape = cnt.shape
            t = cnt.reshape((shape[0],shape[2]))
            max_x,max_y = np.max(t,axis=0)
            min_x,min_y = np.min(t,axis=0)

            if (min_y>=1276-delta) and (max_y<=2097+delta):
                perimeter = cv2.arcLength(cnt,True)

                if perimeter > 100:
                    cv2.drawContours(template_image,[cnt],0,255,3)
                    horizontal_lines.append(cnt)

        horizontal_lines.sort(key = lambda l:l[0][0][1])

        _,vertical_contours = self.__get_lines__(False)

        delta = 400
        for cnt in vertical_contours:
            shape = cnt.shape
            t = cnt.reshape((shape[0],shape[2]))
            max_x,max_y = np.max(t,axis=0)
            min_x,min_y = np.min(t,axis=0)

            interior_line = (min_x >= 563-100) and (max_x <= 3282+100)and(min_y>=1276-delta) and (max_y<=2097+delta)
            through_line = (min_x >= 563-100) and (max_x <= 3282+100) and (min_y < 1276) and(max_y > 2097)

            if interior_line or through_line:

                perimeter = cv2.arcLength(cnt,True)
                if perimeter > 1000:
                    print min_y,max_y
                    # cv2.drawContours(masks,[cnt],0,255,3)
                    vertical_lines.append(cnt)
                    cv2.drawContours(template_image,[cnt],0,255,3)

        vertical_lines.sort(key = lambda l:l[0][0][0])

        # todo - deal with lines that go all the way through the window
        # cv2.imwrite("/home/ggdhines/testtest.jpg",a)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image',a)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return horizontal_lines,vertical_lines

    def __extract_cell__(self,image,h_index,v_index,colour=True):
        """
        :param image: can use an approximate image which is based on a threshold using more global values
        better for determining whether or a cell is empty. If we know that a cell is not empty, we can do thresholding
        based on more local values
        :param h_index:
        :param v_index:
        :return:
        """
        # todo - resizing might be not be necessary
        # todo - DEFINITELY precalculate these
        shape = self.horizontal_grid[h_index].shape
        t = self.horizontal_grid[h_index].reshape((shape[0],shape[2]))
        _,min_y = np.max(t,axis=0)

        shape = self.horizontal_grid[h_index+1].shape
        t = self.horizontal_grid[h_index+1].reshape((shape[0],shape[2]))
        _,max_y = np.min(t,axis=0)

        # repeat for vertical grid lines
        shape = self.vertical_grid[v_index].shape
        t = self.vertical_grid[v_index].reshape((shape[0],shape[2]))
        min_x,_ = np.max(t,axis=0)

        shape = self.vertical_grid[v_index+1].shape
        t = self.vertical_grid[v_index+1].reshape((shape[0],shape[2]))
        max_x,_ = np.min(t,axis=0)

        mask = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
        cv2.drawContours(mask,self.horizontal_grid,h_index,255,-1)
        cv2.drawContours(mask,self.horizontal_grid,h_index+1,255,-1)
        cv2.drawContours(mask,self.vertical_grid,v_index,255,-1)
        cv2.drawContours(mask,self.vertical_grid,v_index+1,255,-1)

        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        for c,h in zip(contours,hier[0]):
            if h[-1] != -1:
                continue

            cv2.drawContours(mask,[c],0,255,-1)

        res = cv2.bitwise_and(mask,image)

        # now go back over and draw in contours in white - any black inside the cell that is on a grid
        # line is mostly likely grid, not ink
        # most of the time this wouldn't make a difference - but be sure
        # todo - double check
        temp_res = res[min_y:max_y+1,min_x:max_x+1]
        most_common_pigment = int(stats.mode(temp_res,axis=None)[0][0])
        # print most_common_pigment
        # print type(most_common_pigment)

        cv2.drawContours(res,self.horizontal_grid,h_index,most_common_pigment,-1)
        cv2.drawContours(res,self.horizontal_grid,h_index+1,most_common_pigment,-1)
        cv2.drawContours(res,self.vertical_grid,v_index,most_common_pigment,-1)
        cv2.drawContours(res,self.vertical_grid,v_index+1,most_common_pigment,-1)

        # now that we have cleaned up this cell, actually zoom into it
        res = res[min_y:max_y+1,min_x:max_x+1]

        # tesseract needs things in colour, so convert

        if colour:
            colour_res = np.zeros((res.shape[0],res.shape[1],3),np.uint8)
            colour_res[:,:,0] = res[:,:]
            colour_res[:,:,1] = res[:,:]
            colour_res[:,:,2] = res[:,:]

            return colour_res
        else:
            return res

    def __process_image__(self,fname):
        image = cv2.imread(fname,0)

        approximate_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,0)

        for h_index in range(len(self.horizontal_grid)-1):
            for v_index in range(len(self.vertical_grid)-1):
                # start by just checking if that cell is empty or not
                cell = self.__extract_cell__(approximate_image,h_index,v_index)

                classifier.tess.set_image(cell)
                classifier.tess.get_utf8_text()

                words = list(classifier.tess.words())
                words_in_cell = [w.text for w in words if w.text is not None]
                conf_in_cell = [w.confidence for w in words if w.text is not None]

                # todo - can we get more detail knowing that the cell is not empty?
                # # is there something in the cell, if so extract that cell in full detail
                #
                # if words != [None]:
                #     cell = self.__extract_cell__(image,h_index,v_index,colour=False)
                #     _,cell = cv2.threshold(cell,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                # tesseract might split up the word into multiple parts - usually a sign that something has
                # gone wrong but we'll see
                word = "".join(words_in_cell)
                if word != "":
                    cv2.imwrite("/home/ggdhines/cell.jpg",cell)
                    overall_confidence = np.mean(conf_in_cell)
                    cell_string = raw_input("Look at cell.jpg - enter its contents: ")

                    # convert to grayscale - the RGB values are all the same
                    digits,minimum_y = extract_digits.extract(cell[:,:,0])

                    cell_list = [c for c in cell_string]

                    print cell_list
                    print len(digits)
                    if len(cell_list) == len(digits):
                        print "adding learning cases"
                        classifier.__add_characters__(cell_list,digits,minimum_y)

                        classifier.__update_tesseract__()

project = ActiveWeather()
project.__process_image__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")