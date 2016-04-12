import numpy as np
import cv2
import os
from subprocess import call
__author__ = 'ggdhines'

spacing = 35

class TesseractUpdate:
    def __init__(self):
        self.boxes = {}
        self.row_pointer = None
        self.column_pointer = None
        self.training_page = None

        self.box_file_entries = []

        if not os.path.exists("/tmp/tessdata"):
            os.makedirs("/tmp/tessdata")

        with open("active_weather.basic.exp0.box","w") as f:
            f.write("")

    def __create_blank_page__(self):
        """
        create a blank page where we'll put labelled characters
        :return:
        """
        self.width = 2508
        self.height = 2480
        # self.height = 4000
        self.training_page = np.zeros((self.height,self.width),dtype=np.uint8)
        self.training_page.fill(255)

        self.row_bitmaps = []
        self.row_characters = []

        self.row_pointer = spacing
        self.column_pointer = spacing

        self.used_height = spacing

    def __next_row(self):
        """
        add each of the saved characters into the row
        :return:
        """
        if self.row_bitmaps != []:
            self.__update_training_image__()
            self.__update_box_entries__()

            # move onto the next row
            row_height = max([b.shape[0] for b in self.row_bitmaps])

            self.row_bitmaps = []
            self.row_characters = []
            self.column_pointer = spacing
            self.row_pointer += spacing + row_height

    def __get_image_height__(self):
        """
        we'll trim the training image to be not much bigger than the text so return the height
        :return:
        """
        if self.row_bitmaps == []:
            return self.row_pointer+spacing
        else:
            row_height = max([b.shape[0] for b in self.row_bitmaps])
            return self.row_pointer+row_height+spacing

    def __update_training_image__(self,save_image=False):
        """
        put the bitmaps into the training image
        :return:
        """
        column_pointer = spacing

        for b in self.row_bitmaps:
            assert isinstance(b,np.ndarray)
            height,width = b.shape

            # row first and then column
            self.training_page[self.row_pointer:self.row_pointer+height,column_pointer:column_pointer+width] = b

            column_pointer += width + spacing
        print([b.shape for b in self.row_bitmaps])

        if save_image:
            cv2.imwrite("active_weather.basic.exp0.tiff",self.training_page)

    def __box_file_flush__(self):
        """
        write out the entries to the box file
        :return:
        """
        # pruned_height = self.__get_image_height__()
        with open("active_weather.basic.exp0.box","w") as f:
            for a,b,c,d,e in self.box_file_entries:
                f.write(str(a)+" "+str(b)+" "+str(c)+" " + str(d) + " " + str(e) + " 0\n")


    def __update_box_entries__(self):
        """
        update the list of box file entries
        :return:
        """
        column_pointer = spacing
        # pruned_height = self.__get_image_height__()

        # with open("/tmp/tessdata/active_weather.lobster.exp0.box","a") as f:
        for char,b in zip(self.row_characters,self.row_bitmaps):
            # calculate the coordinates for the box file
            height,width = b.shape

            self.box_file_entries.append([char,column_pointer,self.height-(self.row_pointer+height-1),column_pointer+width-1,self.height-self.row_pointer])

            column_pointer += width + spacing

    def __add_char__(self,character,bitmap):
        """
        add a new character to our 'training page'
        and flush out the row if we gone too far
        :param character:
        :param additional_y_offset:
        :return:
        """
        char_height,char_width = bitmap.shape

        # do we have too many characters for this row?
        # if so - flush
        if (self.column_pointer+char_width) >= self.width-spacing:
            self.__next_row()

        self.row_bitmaps.append(bitmap)
        self.row_characters.append(character)
        self.column_pointer += char_width + spacing


    # def __add_characters__(self,char_list,image_list,minimum_y):
    #     """
    #     add a whole bunch of characters all at once - while keeping track of what the characters are
    #     so we can add them to the box file
    #     :param char_list:
    #     :param image_list:
    #     :param minimum_y:
    #     :return:
    #     """
    #     top_of_row = min(minimum_y)
    #
    #     for char,img,my in zip(char_list,image_list,minimum_y):
    #         additional_y_offset = my-top_of_row
    #
    #         a,b,c,d = self.__add_char__(img,additional_y_offset)
    #         self.boxes[(a,b,c,d)] = char
    #
    #         self.max_height = max(self.max_height,b)
    #         self.max_width = max(self.max_width,c)
    #
    #     cv2.imwrite("/home/ggdhines/test.jpg",self.training_page)

    # def __write_training__(self):
    #     """
    #     write out the 'training page' and the box file
    #     :return:
    #     """
    #     if not os.path.exists("/tmp/tessdata"):
    #         os.makedirs("/tmp/tessdata")
    #
    #     to_save = self.training_page[0:self.max_height+spacing,0:self.max_width+spacing]
    #
    #     cv2.imwrite("/tmp/tessdata/active_weather.lobster.exp0.tiff",to_save)
    #
    #     with open("/tmp/tessdata/active_weather.lobster.exp0.box","wb") as f:
    #         for (a,b,c,d),char in self.boxes.items():
    #             f.write(char + " " +str(a) + " ")
    #             f.write(str(self.max_height-b+spacing) + " " + str(c) + " " + str(self.max_height-d+spacing) + " 0\n")

    def __update_tesseract__(self):
        """
        actually run the code to update Tesseract
        :return:
        """
        self.__update_training_image__(True)
        self.__box_file_flush__()

        # os.chdir('tessdata')
        call(["tesseract", "active_weather.basic.exp0.tiff", "active_weather.basic.exp0", "nobatch" ,"box.train"])
        # raw_input("hello world")
        call(["unicharset_extractor", "active_weather.basic.exp0.box"])
        # raw_input("goodbye")

        with open("font_properties","w") as f:
            f.write("basic 0 0 0 0 0\n")

        os.system("shapeclustering -F font_properties -U unicharset active_weather.basic.exp0.tr")

        os.system("mftraining -F font_properties -U unicharset -O active_weather.unicharset active_weather.basic.exp0.tr")
        os.system("cntraining active_weather.basic.exp0.tr")

        os.system("mv inttemp active_weather.inttemp")
        os.system("mv normproto active_weather.normproto")
        os.system("mv pffmtable active_weather.pffmtable")
        os.system("mv shapetable active_weather.shapetable")
        os.system("combine_tessdata active_weather.")

        os.system("mv active_weather.basic.* /tmp/tessdata/")
        os.system("mv active_weather.inttemp /tmp/tessdata/")
        os.system("mv active_weather.normproto /tmp/tessdata/")
        os.system("mv active_weather.pffmtable /tmp/tessdata/")
        os.system("mv active_weather.shapetable /tmp/tessdata/")
        os.system("mv active_weather.traineddata /tmp/tessdata/")
        os.system("mv active_weather.unicharset /tmp/tessdata/")
        os.system("mv font_properties /tmp/tessdata/")
