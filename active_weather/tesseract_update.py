import numpy as np
import cv2
import os
from subprocess import call
import matplotlib.pyplot as plt
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

        # with open("active_weather.basic.exp0.box","w") as f:
        #     f.write("")

        self.box_count = 0

        self.character_heights = []

    def __create_blank_page__(self):
        """
        create a blank page where we'll put labelled characters
        :return:
        """
        with open("active_weather.basic.exp"+str(self.box_count)+".box","w") as f:
            f.write("")

        self.width = 2508
        # self.height = 200
        self.height = 4000
        self.training_page = np.zeros((self.height,self.width),dtype=np.uint8)
        self.training_page.fill(255)

        self.row_bitmaps = []
        self.row_characters = []

        self.row_pointer = spacing
        self.column_pointer = spacing


        # self.__box_file_flush__()
        self.box_file_entries = []
        self.used_height = spacing

    # def __next_image__(self):
    #     """
    #     add each of the saved characters into the row
    #     :return:
    #     """
    #     # if self.row_bitmaps != []:
    #     #     # self.__update_training_image__()
    #     #     # self.__update_box_entries__()
    #     #
    #     #     # move onto the next row
    #     #     row_height = max([b.shape[0] for b in self.row_bitmaps])
    #     #
    #     #
    #     #     # self.__update_box_entries__()
    #     #     # self.__box_file_flush__()
    #     self.__update_training_image__(save_image=True)
    #     call(["tesseract", "active_weather.basic.exp"+str(self.box_count)+".tiff", "active_weather.basic.exp"+str(self.box_count), "nobatch" ,"box.train"])
    #
    #     self.box_count += 1
    #
    #     self.row_bitmaps = []
    #     self.row_characters = []
    #     self.column_pointer = spacing
    #
    #     self.training_page = None
    #     # self.row_pointer += spacing + row_height

    # def __get_image_height__(self):
    #     """
    #     we'll trim the training image to be not much bigger than the text so return the height
    #     :return:
    #     """
    #     if self.row_bitmaps == []:
    #         return self.row_pointer+spacing
    #     else:
    #         row_height = max([b.shape[0] for b in self.row_bitmaps])
    #         return self.row_pointer+row_height+spacing

    # def __update_training_image__(self,save_image=False):
    #     """
    #     put the bitmaps into the training image
    #     :return:
    #     """
    #     column_pointer = spacing
    #
    #     for b in self.row_bitmaps:
    #         assert isinstance(b,np.ndarray)
    #         height,width = b.shape
    #
    #         # row first and then column
    #         self.training_page[self.row_pointer:self.row_pointer+height,column_pointer:column_pointer+width] = b
    #
    #         column_pointer += width + spacing
    #     # assert([b.shape for b in self.row_bitmaps] != [])
    #     # print([b.shape for b in self.row_bitmaps])
    #
    #     if save_image:
    #         cv2.imwrite("active_weather.basic.exp"+str(self.box_count)+".tiff",self.training_page)

    # def __box_file_flush__(self):
    #     """
    #     write out the entries to the box file
    #     :return:
    #     """
    #     # pruned_height = self.__get_image_height__()
    #     with open("active_weather.basic.exp"+str(self.box_count)+".box","w") as f:
    #         for a,b,c,d,e in self.box_file_entries:
    #             f.write(str(a)+" "+str(b)+" "+str(c)+" " + str(d) + " " + str(e) + " 0\n")


    # def __box_file_update__(self):
    #     """
    #     update the list of box file entries
    #     :return:
    #     """
    #     # print("updating")
    #
    #     # pruned_height = self.__get_image_height__()
    #
    #     # with open("/tmp/tessdata/active_weather.lobster.exp0.box","a") as f:
    #     with open("active_weather.basic.exp"+str(self.box_count)+".box","a") as f:
    #         # for char,b in zip(self.row_characters,self.row_bitmaps):
    #             # calculate the coordinates for the box file
    #         b = self.row_bitmaps[-1]
    #         char = self.row_characters[-1]
    #         height,width = b.shape
    #
    #         a,b,c,d,e = char,self.column_pointer,self.height-(self.row_pointer+height-1),self.column_pointer+width-1,self.height-self.row_pointer
    #         # self.box_file_entries.append([char,column_pointer,self.height-(self.row_pointer+height-1),column_pointer+width-1,self.height-self.row_pointer])
    #         f.write(str(a)+" "+str(b)+" "+str(c)+" " + str(d) + " " + str(e) + " 0\n")
    #
    #         self.column_pointer += width + spacing
    #     # print("active_weather.basic.exp"+str(self.box_count)+".box")
    #     # assert False

    def __write_out_row__(self):
        """
        put the bitmaps into the training image
        :return:
        """
        column_pointer = spacing

        row_height = np.max([b.shape[0] for b in self.row_bitmaps])

        with open("active_weather.basic.exp"+str(self.box_count)+".box","a") as f:
            for char,b in zip(self.row_characters,self.row_bitmaps):
                assert isinstance(b, np.ndarray)
                height, width = b.shape

                # row first and then column
                additional_height = row_height-height

                self.training_page[self.row_pointer+additional_height:self.row_pointer + height+additional_height, column_pointer:column_pointer + width] = b
                a, b, c, d, e = char, column_pointer, self.height - (self.row_pointer + height + additional_height), column_pointer + width, self.height - (self.row_pointer+additional_height)
                f.write(str(a) + " " + str(b) + " " + str(c+1) + " " + str(d-1) + " " + str(e) + " 0\n")

                column_pointer += width + spacing


        self.row_pointer += spacing + row_height
        self.column_pointer = spacing

        self.row_bitmaps = []
        self.row_characters = []

    def __add_char__(self,character,bitmap):
        """
        add a new character to our 'training page'
        and flush out the row if we gone too far
        :param character:
        :param additional_y_offset:
        :return:
        """
        # if self.training_page is None:
        #     self.__create_blank_page__()

        char_height,char_width = bitmap.shape

        # do we have too many characters for this row?
        # if so - flush
        if (self.column_pointer+char_width) >= self.width-spacing:
            self.__write_out_row__()

        # self.character_heights.append(bitmap.shape[0])


        self.row_bitmaps.append(bitmap)
        self.row_characters.append(character)
        self.column_pointer += char_width + spacing

        # self.__box_file_update__()

    def __update_tesseract__(self):
        """
        actually run the code to update Tesseract
        :return:
        """
        if self.row_bitmaps != []:
            self.__write_out_row__()
        cv2.imwrite("active_weather.basic.exp" + str(self.box_count) + ".tiff", self.training_page)
        call(["tesseract", "active_weather.basic.exp0.tiff", "active_weather.basic.exp0", "nobatch", "box.train"])

        with open("font_properties","w") as f:
            f.write("basic 0 0 0 0 0\n")

        call(["unicharset_extractor", "*.box"])

        assert False

        os.system("shapeclustering -F font_properties -U unicharset active_weather.basic.exp0.tr")

        os.system("mftraining -F font_properties -U unicharset -O active_weather.unicharset active_weather.basic.exp0.tr")
        os.system("cntraining active_weather.basic.exp0.tr")

        os.system("mv inttemp active_weather.inttemp")
        os.system("mv normproto active_weather.normproto")
        os.system("mv pffmtable active_weather.pffmtable")
        os.system("mv shapetable active_weather.shapetable")
        os.system("combine_tessdata active_weather.")

        # os.system("mv active_weather.basic.* /tmp/tessdata/")
        # os.system("mv active_weather.inttemp /tmp/tessdata/")
        # os.system("mv active_weather.normproto /tmp/tessdata/")
        # os.system("mv active_weather.pffmtable /tmp/tessdata/")
        # os.system("mv active_weather.shapetable /tmp/tessdata/")
        # os.system("mv active_weather.traineddata /tmp/tessdata/")
        # os.system("mv active_weather.unicharset /tmp/tessdata/")
        # os.system("mv font_properties /tmp/tessdata/")
