#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import glob
import matplotlib
# matplotlib.use('WXAgg')
# import matplotlib.pyplot as plt
import database_connection
import extract_digits
from scipy import stats
import tesseract_font
from subprocess import call
import Image
import pytesseract
import os
horizontal = []
import hocr
import tesserpy

# just for size reference
reference_subject = "Bear-AG-29-1940-0019"
reference_image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"+reference_subject+".JPG")
refer_shape = reference_image.shape

class ActiveWeather:
    def __init__(self):
        self.cass_db = database_connection.Database()
        print("connected to the db")
        # self.__get_grid__()
        self.horizontal_grid = self.cass_db.__get_horizontal_lines__(reference_subject,0)
        self.vertical_grid = self.cass_db.__get_vertical_lines__(reference_subject,0)
        # self.horizontal_grid,self.vertical_grid = self.__get_grid__()

        self.region = 0

        self.classifier = tesseract_font.ActiveTess()



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
        self.cass_db.__add_horizontal_lines__(reference_subject,0,horizontal_lines)
        self.cass_db.__add_vertical_lines__(reference_subject,0,vertical_lines)

        return horizontal_lines,vertical_lines



    def __process_row__(self,fname,row_index):
        image = cv2.imread(fname,0)
        # most_common_pigment = int(stats.median(image,axis=None)[0][0])
        most_common_pigment = np.median(image)
        print(most_common_pigment)
        min_x,max_x,min_y,max_y,mask1,mask2 = self.__create_masks__(row_index)

        row_contents = cv2.bitwise_and(image,mask1)
        background = np.where(mask2>0)
        row_contents[background] = most_common_pigment

        # remove the vertical rows
        for v in self.vertical_grid:
            cv2.drawContours(row_contents,[v],0,255,-1)

        row = row_contents[min_y:max_y+1,min_x:max_x+1]

        ret2,th2 = cv2.threshold(row,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite("/home/ggdhines/row.jpg",th2)


        # # row_ = cv2.adaptiveThreshold(row,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,2)
        # ret,row_ = cv2.threshold(row,most_common_pigment+5,255,cv2.THRESH_BINARY)
        # cv2.imwrite("/home/ggdhines/row_.jpg",row_)

        _,contour, hier = cv2.findContours(th2.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt,h_ in zip(contour,hier[0]):

            x,y,w,h = cv2.boundingRect(cnt)

            # print(h)
            # print(w/h)
            # print("")
            if (h < 10):
                print(h)
                cv2.drawContours(th2,[cnt],0,255,-1)


        cv2.imwrite("/home/ggdhines/row2.jpg",th2)


        row_colour = np.zeros((th2.shape[0],th2.shape[1],3),np.uint8)
        row_colour[:,:,0] = th2
        row_colour[:,:,1] = th2
        row_colour[:,:,2] = th2

        tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/",language="eng")
        tess.tessedit_char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890."
        tess.set_image(row_colour)
        tess.get_utf8_text()
        words = list(tess.words())
        words_in_cell = [w.text for w in words if w.text is not None]
        print(words_in_cell)
        conf_in_cell = [w.confidence for w in words if w.text is not None]
        print(conf_in_cell)

        # words,confidence = hocr.scan()
        # for w,c in zip(words,confidence):
        #     print((w,c))
        raw_input("check row.jpg")

    def __region_mask__(self):
        mask = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
        cv2.drawContours(mask,self.horizontal_grid,0,255,-1)
        cv2.drawContours(mask,self.horizontal_grid,len(self.horizontal_grid)-2,255,-1)
        cv2.drawContours(mask,self.vertical_grid,0,255,-1)
        cv2.drawContours(mask,self.vertical_grid,len(self.vertical_grid)-1,255,-1)

        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        mask2 = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)

        for c,h in zip(contours,hier[0]):
            if h[-1] == -1:
                continue

            cv2.drawContours(mask2,[c],0,255,-1)


        cv2.imwrite("/home/ggdhines/mask.jpg",mask)
        return mask2

    def __process_region__(self,fname):
        image = cv2.imread(fname,0)
        mask = self.__region_mask__()

        masked_image = cv2.bitwise_and(image,mask)
        most_common_pigment = np.median(image)
        # cv2.drawContours(masked_image,self.horizontal_grid,-1,most_common_pigment,-1)
        # cv2.drawContours(masked_image,self.vertical_grid,-1,most_common_pigment,-1)

        cv2.drawContours(masked_image,self.horizontal_grid,-1,most_common_pigment,-1)
        cv2.drawContours(masked_image,self.vertical_grid,-1,most_common_pigment,-1)

        t = self.horizontal_grid[0]#.reshape((shape[0],shape[2]))
        _,min_y = np.min(t,axis=0)

        t = self.horizontal_grid[-2]#.reshape((shape[0],shape[2]))
        _,max_y = np.max(t,axis=0)

        # repeat for vertical grid lines
        # shape = self.vertical_grid[v_index].shape
        t = self.vertical_grid[0]#.reshape((shape[0],shape[2]))
        min_x,_ = np.max(t,axis=0)

        t = self.vertical_grid[-1]#.reshape((shape[0],shape[2]))
        max_x,_ = np.min(t,axis=0)

        region = masked_image[min_y:max_y+1,min_x:max_x+1]
        # ret2,th2 = cv2.threshold(region,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # th2 = cv2.adaptiveThreshold(region,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,401,2)
        cv2.imwrite("/home/ggdhines/step1.jpg",region)

        mask3 = np.zeros(image.shape,np.uint8)
        mask3[min_y:max_y+1,min_x:max_x+1] = 255
        mask4 = cv2.bitwise_xor(mask3,mask)

        mask4_t = np.where(mask4>0)
        masked_image[mask4_t] = most_common_pigment
        region = masked_image[min_y:max_y+1,min_x:max_x+1]
        cv2.imwrite("/home/ggdhines/step2.jpg",region)

        ret,thresh1 = cv2.threshold(region,190,255,cv2.THRESH_BINARY)
        cv2.imwrite("/home/ggdhines/step3.jpg",thresh1)

        _,contours, hier = cv2.findContours(thresh1.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            print(perimeter)
            if perimeter <= 50:
                cv2.drawContours(thresh1,[cnt],0,255,-1)

        cv2.imwrite("/home/ggdhines/step4.jpg",thresh1)


    def __row_mask__(self,row_index):
        """
        creates two masks - for one extracting the cell contents
        the other for allowing a proper rectangle to be formed - the second mask will be used
        for setting the background colour
        :param fname:
        :param row_index:
        :return:
        """
        # image = cv2.imread(fname,0)
        # most_common_pigment = int(stats.mode(image,axis=None)[0][0])
        # image = cv2.drawContours(image,self.horizontal_grid,-1,most_common_pigment,-1)
        # image = cv2.drawContours(image,self.vertical_grid,-1,most_common_pigment,-1)


        mask = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
        cv2.drawContours(mask,self.horizontal_grid,row_index,255,-1)
        cv2.drawContours(mask,self.horizontal_grid,row_index+1,255,-1)
        cv2.drawContours(mask,self.vertical_grid,0,255,-1)
        cv2.drawContours(mask,self.vertical_grid,len(self.vertical_grid)-1,255,-1)
        cv2.imwrite("/home/ggdhines/mask.jpg",mask)

        # _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        mask2 = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
        for c,h in zip(contours,hier[0]):
            if h[-1] == -1:
                continue

            cv2.drawContours(mask2,[c],0,255,-1)

        t = self.horizontal_grid[row_index]#.reshape((shape[0],shape[2]))
        _,min_y = np.min(t,axis=0)

        t = self.horizontal_grid[row_index+1]#.reshape((shape[0],shape[2]))
        _,max_y = np.max(t,axis=0)

        # repeat for vertical grid lines
        # shape = self.vertical_grid[v_index].shape
        t = self.vertical_grid[0]#.reshape((shape[0],shape[2]))
        min_x,_ = np.max(t,axis=0)

        t = self.vertical_grid[-1]#.reshape((shape[0],shape[2]))
        max_x,_ = np.min(t,axis=0)

        mask3 = np.zeros((refer_shape[0],refer_shape[1]),np.uint8)
        mask3[min_y:max_y+1,min_x:max_x+1] = 255
        cv2.imwrite("/home/ggdhines/mask3.jpg",mask3)

        mask4 = cv2.bitwise_xor(mask3,mask2)
        cv2.imwrite("/home/ggdhines/mask4.jpg",mask4)

        return min_x,max_x,min_y,max_y,mask2,mask4
        # assert False
        #
        #
        # # print((min_x,min_y,max_x,max_y))
        #
        # row = image[min_y:max_y,min_x:max_x]
        #
        # # colour_image = np.zeros((row.shape[0],row.shape[1],3),np.uint8)
        # # colour_image[:,:,0] = row
        # # colour_image[:,:,1] = row
        # # colour_image[:,:,2] = row
        #
        # cv2.imwrite("/home/ggdhines/output.jpg",row)
        #
        # words,confidence = hocr.scan()
        # print(words)



    def __extract_cell__(self,image,h_index,v_index,colour=True):
        """
        :param image: can use an approximate image which is based on a threshold using more global values
        better for determining whether or a cell is empty. If we know that a cell is not empty, we can do thresholding
        based on more local values
        :param h_index:
        :param v_index:
        :return:
        """
        # todo - DEFINITELY precalculate these
        # shape = self.horizontal_grid[h_index].shape
        t = self.horizontal_grid[h_index]#.reshape((shape[0],shape[2]))
        _,min_y = np.max(t,axis=0)

        shape = self.horizontal_grid[h_index+1].shape
        t = self.horizontal_grid[h_index+1]#.reshape((shape[0],shape[2]))
        _,max_y = np.min(t,axis=0)

        # repeat for vertical grid lines
        # shape = self.vertical_grid[v_index].shape
        t = self.vertical_grid[v_index]#.reshape((shape[0],shape[2]))
        min_x,_ = np.max(t,axis=0)

        # shape = self.vertical_grid[v_index+1].shape
        t = self.vertical_grid[v_index+1]#.reshape((shape[0],shape[2]))
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

        boundary = [min_y,max_y+1,min_x,max_x+1]

        if colour:
            colour_res = np.zeros((res.shape[0],res.shape[1],3),np.uint8)
            colour_res[:,:,0] = res[:,:]
            colour_res[:,:,1] = res[:,:]
            colour_res[:,:,2] = res[:,:]

            return colour_res,boundary
        else:
            return res,boundary

    def __process_image__(self,fname):
        self.__process_region__(fname)
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
    project.__process_image__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__remove_template__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")