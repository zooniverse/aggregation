from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from subprocess import call
import tesserpy

# call(["convert","weather.basic.exp0.pdf","/home/ggdhines/tessdata/active.basic.exp0.tiff"])

spacing = 35


def create_blank_sheet_(character_dict):
    assert isinstance(character_dict,dict)

    num_samples = [len(character_dict[c]) if c in character_dict else 0 for c in characters]

    width_list = []
    height_list = []

    for c in characters:
        if c in character_dict:
            width,height = character_dict[c][0].shape
            width_list.append(width)
            height_list.append(height)
        else:
            width_list.append(0)
            height_list.append(0)

    needed_width = [n*w+spacing*n for (n,w) in zip(num_samples,width_list)]
    overall_width = max(needed_width)

    overall_height = sum(height_list) + spacing*len(characters)

class ActiveTess:
    def __init__(self):
        self.example_page = None

        self.__create_blank_page__()

        # self.tess = tesserpy.Tesseract("/home/ggdhines/PycharmProjects/reduction/active_weather/tessdata/", language="active_weather")
        self.tess = tesserpy.Tesseract("/home/ggdhines/github/tessdata/",language="eng")
        self.tess.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_BLOCK
        self.tess.tessedit_char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.abcdefghijklmnopqrstuvwxyz"

        self.boxes = dict()

        self.max_height = 0
        self.max_width = 0



    def __process_image__(self,image):
        # self.tess.tessedit_pageseg_mode = tesserpy.PSM_SINGLE_CHAR
        self.tess.set_image(image)
        self.tess.get_utf8_text()
        text = []
        confidences = []
        boxes = []
        for word in self.tess.words():
            # bb = word.bounding_box
            # print("{}\t{}\tt:{}; l:{}; r:{}; b:{}".format(word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
            confidences.append(word.confidence)
            text.append(word.text)
            boxes.append(word.bounding_box)

        return boxes,text

    def __is_blank__(self,fname):
        image = cv2.imread(fname)
        assert image is not None
        self.tess.set_image(image)

        self.tess.get_utf8_text()
        words = [w.text for w in self.tess.words()]
        return words == [None]

    def __process_column__(self,fname):
        image = cv2.imread(fname)
        self.tess.set_image(image)

        self.tess.get_utf8_text()

        text = []
        confidences = []
        for word in self.tess.words():
            # bb = word.bounding_box
            # print("{}\t{}\tt:{}; l:{}; r:{}; b:{}".format(word.text, word.confidence, bb.top, bb.left, bb.right, bb.bottom))
            confidences.append(word.confidence)
            text.append(word.text)

        return text,confidences

    def __create_blank_page__(self):
        self.example_page = np.zeros((2480,2508),dtype=np.uint8)
        self.example_page.fill(255)

        self.curr_voffset = spacing
        self.curr_hoffset = spacing

    def __add_char__(self,character):
        assert isinstance(character,np.ndarray)
        height,width = character.shape

        # figure out where on the page we want to place
        h_lb = self.curr_hoffset
        h_ub = h_lb + height

        w_lb = self.curr_voffset
        w_ub = w_lb + width

        self.example_page[h_lb:h_ub,w_lb:w_ub] = character

        self.curr_voffset += width + spacing

        return w_lb,h_ub,w_ub,h_lb




    def __add_characters__(self,char_list,image_list):
        # top_of_row = min(minimum_y)

        for char,img in zip(char_list,image_list):
            additional_y_offset = 0#my-top_of_row

            a,b,c,d = self.__add_char__(img)
            self.boxes[(a,b,c,d)] = char

            self.max_height = max(self.max_height,b)
            self.max_width = max(self.max_width,c)

        cv2.imwrite("/home/ggdhines/test.jpg",self.example_page)

    def __write_out_box_file__(self):
        if not os.path.exists("/tmp/tessdata"):
            os.makedirs("/tmp/tessdata")

        to_save = self.example_page[0:self.max_height+spacing,0:self.max_width+spacing]

        cv2.imwrite("/tmp/tessdata/active_weather.lobster.exp0.tiff",to_save)

        with open("/tmp/tessdata/active_weather.lobster.exp0.box","wb") as f:
            for (a,b,c,d),char in self.boxes.items():
                f.write(char + " " +str(a) + " ")
                f.write(str(self.max_height-b+spacing) + " " + str(c) + " " + str(self.max_height-d+spacing) + " 0\n")

    def __update_tesseract__(self):
        self.__write_out_box_file__()

        os.chdir('/tmp/tessdata')
        call(["tesseract", "active_weather.lobster.exp0.tiff", "active_weather.lobster.exp0", "nobatch" ,"box.train"])
        # call(["unicharset_extractor", "active_weather.lobster.exp0.box"])
        call(["unicharset_extractor", "*.box"])

        with open("/tmp/tessdata/font_properties","w") as f:
            f.write("lobster 0 0 0 0 0\n")

        os.system("shapeclustering -F font_properties -U unicharset active_weather.lobster.exp0.tr")
        # "mftraining -F font_properties -U unicharset -O active_weather.unicharset active_weather.lobster.exp0.tr"
        os.system("mftraining -F font_properties -U unicharset -O active_weather.unicharset active_weather.lobster.exp0.tr")
        os.system("cntraining active_weather.lobster.exp0.tr")

        os.system("mv inttemp active_weather.inttemp")
        os.system("mv normproto active_weather.normproto")
        os.system("mv pffmtable active_weather.pffmtable")
        os.system("mv shapetable active_weather.shapetable")
        os.system("combine_tessdata active_weather.")


# f.write(c)
# f.write(" "+str(width_offset)+" "+str(overall_height - height_offset-height))
# f.write(" "+str(width_offset+width)+" "+str(overall_height - height_offset) + " 0\n")

if __name__ == "__main__":
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890."
    characters = [c for c in characters]
    print(characters)

    image = cv2.imread("weather.basic.exp0.tif")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,bw_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    pixel_points = np.where(bw_image<255)

    # plt.plot(pixel_points[1],pixel_points[0],".")
    # plt.show()

    X = np.asarray(zip(pixel_points[1],pixel_points[0]))

    db = DBSCAN(eps=3, min_samples=4).fit(X)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    clusters = []

    height_list = []
    width_list = []

    print(n_clusters_)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # start by just ordering the clusters so we can match them up to their characters
    for k, col in zip(unique_labels, colors):

        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        clusters.append(xy)

    clusters.sort(key = lambda c:np.mean(c[:,0]))

    for cluster_index,xy in enumerate(clusters):
        min_x,min_y = np.min(xy,axis=0)
        max_x,max_y = np.max(xy,axis=0)



        height = max_y-min_y+1
        width = max_x-min_x +1

        print(min_x,min_y,height,width)

        height_list.append(height)
        width_list.append(width)

        xy[:,0] -= min_x
        xy[:,1] -= min_y

        clusters[cluster_index] = xy



    overall_width = 10*max(width_list) + spacing*11 + 150
    # height_list[-2] for book ends at the bottom
    overall_height = sum(height_list) + spacing*len(characters) + height_list[-2]

    print(overall_height,overall_width)

    template_image = np.zeros((overall_height,overall_width),np.uint8)
    template_image.fill(255)

    # cv2.imwrite("/home/ggdhines/fonts.jpg",template_image)

    print(template_image)
    print(template_image.shape)
    height_offset = 0

    with open("/home/ggdhines/tessdata/active_weather.lobster.exp0.box","w") as f:
        for i,c in enumerate(characters[:-1]):
            # char_mask = np.zeros((height_list[i],width_list[i]),np.uint8)
            char_mask = np.zeros((height_list[i],width_list[i]),np.uint8)
            char_mask.fill(255)

            # todo - do this better
            for x,y in clusters[i]:
                char_mask[y,x] = 0

            height = height_list[i]
            width = width_list[i]

            print(char_mask.shape)

            width_offset = spacing
            for column in range(10):
                template_image[height_offset:height_offset+height,width_offset:width_offset+width] = char_mask

                f.write(c)
                f.write(" "+str(width_offset)+" "+str(overall_height - height_offset-height))
                f.write(" "+str(width_offset+width)+" "+str(overall_height - height_offset) + " 0\n")

                width_offset += width + spacing

            height_offset += height + spacing

        #
        #
        # # since "."s are usually discounted as noise - need to wrap something around them to get tesseract to notice
        i = len(characters)-1
        c = characters[-1]

        b_e_height = height_list[i-1]
        b_e_width = width_list[i-1]

        book_end_mask = np.zeros((b_e_height,b_e_width),np.uint8)
        book_end_mask.fill(255)

        for x,y in clusters[i-1]:
            book_end_mask[y,x] = 0




        width_offset = spacing
        template_image[height_offset:height_offset+b_e_height,width_offset:width_offset+b_e_width] = book_end_mask


        f.write("0")
        f.write(" "+str(width_offset)+" "+str(overall_height - height_offset - b_e_height))
        f.write(" "+str(width_offset+b_e_width)+" "+str(overall_height - height_offset) + " 0\n")

        width_offset += spacing+b_e_width

        # char_mask = np.zeros((height_list[i],width_list[i]),np.uint8)
        char_mask = np.zeros((height_list[i],width_list[i]),np.uint8)
        char_mask.fill(255)

        # todo - do this better
        for x,y in clusters[i]:
            char_mask[y,x] = 0

        height = height_list[i]
        width = width_list[i]

        print(width)

        extra_height = b_e_height - height

        width_offset += b_e_width + spacing


        for column2 in range(10):
            lb_height = height_offset + extra_height
            ub_height = lb_height + height
            template_image[lb_height:ub_height,width_offset:width_offset+width] = char_mask
        #
            f.write(c)
            f.write(" "+str(width_offset)+" "+str(overall_height - ub_height))
            f.write(" "+str(width_offset+width)+" "+str(overall_height - lb_height) + " 0\n")
        #
            width_offset += width + spacing
        #
        template_image[height_offset:height_offset+b_e_height,width_offset:width_offset+b_e_width] = book_end_mask

        f.write("0")
        f.write(" "+str(width_offset)+" "+str(overall_height - height_offset - b_e_height))
        f.write(" "+str(width_offset+b_e_width)+" "+str(overall_height - height_offset) + " 0\n")


    cv2.imwrite("/home/ggdhines/tessdata/active_weather.lobster.exp0.tiff",template_image)

    os.chdir('/home/ggdhines/tessdata/')
    call(["tesseract", "active_weather.lobster.exp0.tiff", "active_weather.lobster.exp0", "nobatch" ,"box.train"])
    call(["unicharset_extractor", "active_weather.lobster.exp0.box"])

    with open("/home/ggdhines/tessdata/font_properties","w") as f:
        f.write("lobster 0 0 0 0 0\n")

    os.system("shapeclustering -F font_properties -U unicharset active_weather.lobster.exp0.tr")
    # "mftraining -F font_properties -U unicharset -O active_weather.unicharset active_weather.lobster.exp0.tr"
    os.system("mftraining -F font_properties -U unicharset -O active_weather.unicharset active_weather.lobster.exp0.tr")
    os.system("cntraining active_weather.lobster.exp0.tr")

    os.system("mv inttemp active_weather.inttemp")
    os.system("mv normproto active_weather.normproto")
    os.system("mv pffmtable active_weather.pffmtable")
    os.system("mv shapetable active_weather.shapetable")
    os.system("combine_tessdata active_weather.")
