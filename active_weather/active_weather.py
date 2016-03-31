from __future__ import print_function
import matplotlib.pyplot as plt
import cv2
import preprocessing
__author__ = 'ggdhines'


class ActiveWeather:
    def __init__(self):
        self.regions =  [(559,3282,1276,2097)]

    def __extract_region__(self,fname,region_id = 0):
        img = cv2.imread(fname)
        region = self.regions[region_id]
        sub_image = img[region[2]:region[3],region[0]:region[1]]

        return sub_image

project = ActiveWeather()
fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0005.JPG"
img = project.__extract_region__(fname)
id_ = fname.split("/")[-1][:-4]


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
masked_image = preprocessing.__mask_lines__(gray)
transcriptions = preprocessing.__ocr_image__(masked_image)
transcriptions_in_cells = preprocessing.__place_in_cell__(transcriptions,gray,id_)