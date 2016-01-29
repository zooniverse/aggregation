import os
# from template_align import align
import numpy as np
import Image
# from extract_digits import extract
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
# import cv2
# import subprocess

image_directory = "/home/ggdhines/Databases/old_weather/pruned_cases/"
template_image = "Bear-AG-29-1941-0557.JPG"


# cell_columns = [(496,698),(698,805),(805,874),(1051,1234),(1405,1508),(1508,1719),(1719,1816),(1816,1927),(1927,2032),(2032,2134),(2733,2863),(2863,2971),(2971,3133)]
# cell_rows = [(1267,1370),(1370,1428),(1428,1488),(1488,1547),(1547,1606),(1606,1665),(1665,1723),(1723,1781),(1781,1840),(1840,1899),(1899,1957),(1957,2016)]

cell_columns = [(713,821),(821,890),(1067,1252),(1527,1739),(1739,1837),(1837,1949),(1949,2053),(2053,2156)] #(510,713),
cell_rows = [(1226,1320),(1320,1377)]

# s3://zooniverse-static/old-weather-2015/War_in_the_Arctic/Greenland_Patrol/Navy/Bear_AG-29_/

log_pages = list(os.listdir(image_directory))

for f_count,f_name in enumerate(log_pages):
    if not f_name.endswith(".JPG"):
        continue

    image_file = cbook.get_sample_data(image_directory+f_name)
    image = plt.imread(image_file)

    p_index = f_name.rfind(".")
    base_fname = f_name[:p_index]

    for row_index,(row_lb,row_ub) in enumerate(cell_rows):
        for column_index,(column_lb,column_ub) in enumerate(cell_columns):
            offset = 8
            r = range(row_lb-offset,row_ub+offset)
            c = range(column_lb-offset,column_ub+offset)

            # print column_lb,column_ub

            # fig, ax = plt.subplots()
            sub_image = image[np.ix_(r, c)]
            # print sub_image.shape
            # im = ax.imshow(sub_image)
            # plt.plot([column_lb,column_lb,column_ub,column_ub,column_lb],[row_lb,row_ub,row_ub,row_lb,row_lb],color="red")
            # plt.savefig("/tmp/aa.png",dpi=500)
            # plt.show()
            # # assert False
            #
            #
            img = Image.fromarray(sub_image, 'RGB')
            # img.convert("L")
            img.save("/home/ggdhines/Databases/old_weather/cells/"+base_fname+"_"+str(row_index)+"_"+str(column_index)+".png")
            #
            #
            # # extract(sub_image)
            # raw_input("enter something")
