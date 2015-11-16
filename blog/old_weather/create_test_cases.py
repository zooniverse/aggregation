import os
import numpy as np
import Image
# from extract_digits import extract
import matplotlib.pyplot as plt
import subprocess

import signal

def sig_handler(signum, frame):
    print "seg fault!!"
    raise

signal.signal(signal.SIGSEGV, sig_handler)

image_directory = "/home/ggdhines/Databases/old_weather/test_cases/"
template_image = "Bear-AG-29-1941-0557.JPG"


# cell_columns = [(496,698),(698,805),(805,874),(1051,1234),(1405,1508),(1508,1719),(1719,1816),(1816,1927),(1927,2032),(2032,2134),(2733,2863),(2863,2971),(2971,3133)]
# cell_rows = [(1267,1370),(1370,1428),(1428,1488),(1488,1547),(1547,1606),(1606,1665),(1665,1723),(1723,1781),(1781,1840),(1840,1899),(1899,1957),(1957,2016)]

cell_columns = [(510,713),(713,821),(821,890),(1219,1252),(1527,1739),(1739,1837),(1837,1949),(1949,2053),(2053,2156)]
cell_rows = [(1226,1320),(1320,1377)]

# s3://zooniverse-static/old-weather-2015/War_in_the_Arctic/Greenland_Patrol/Navy/Bear_AG-29_/

log_pages = list(os.listdir(image_directory))

# for f_count,f_name in enumerate():
f_count = 0
while f_count < min(100,len(log_pages)):
    f_name = log_pages[f_count]
    if f_name.endswith(".JPG"):
        print image_directory+f_name

        # we want to map these new image back to the template image
        # this way we can use the template coordinates without having to do any further transformations
        # image = align(image_directory+template_image,image_directory+f_name)
        # image = align(image_directory+f_name,image_directory+template_image)
        try:
            t = subprocess.check_output(["python","template_align.py",image_directory+f_name,image_directory+template_image])
            # print type(t)
            print t
            f_count += 1
        except subprocess.CalledProcessError:
            print "segfault"

        continue
        # cv2.imwrite("/home/ggdhines/t1.png",image)

        # # cv2.imwrite('/home/ggdhines/messigray.png',image)
        # # break
        # # print type(image)
        # # sub_image = image[162:238,274:326,:]

        for column_lb,column_ub in cell_columns:
            for row_lb,row_ub in cell_rows:

                # image_file = cbook.get_sample_data(base_directory + "/Databases/penguins/images/"+object_id+".JPG")
                # image = plt.imread(image_file)
                # fig, ax = plt.subplots()
                # im = ax.imshow(image)
                # plt.plot([column_lb,column_lb,column_ub,column_ub,column_lb],[row_lb,row_ub,row_ub,row_lb,row_lb],color="red")
                # plt.savefig("/tmp/aa.png",dpi=500)
                # assert False
                # cv2.imwrite("/tmp/aa__.png",image)

                offset = 8
                r = range(row_lb-offset,row_ub+offset)
                c = range(column_lb-offset,column_ub+offset)

                fig, ax = plt.subplots()
                sub_image = image[np.ix_(r, c)]
                # print sub_image.shape
                # im = ax.imshow(sub_image)
                # plt.plot([column_lb,column_lb,column_ub,column_ub,column_lb],[row_lb,row_ub,row_ub,row_lb,row_lb],color="red")
                # plt.savefig("/tmp/aa.png",dpi=500)
                # assert False


                img = Image.fromarray(sub_image, 'RGB')
                img.save('/home/ggdhines/a.png')


                # extract(sub_image)
                raw_input("enter something")