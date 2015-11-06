import os
from template_align import align
import numpy as np
import Image
import cv2
from extract_digits import extract

image_directory = "/home/ggdhines/Databases/old_weather/images/"
template_image = "eastwind-wag-279-1946_0031-0.JPG"

for f_count,f_name in enumerate(os.listdir(image_directory)):
    if f_count == 20:
        break
    if f_name.endswith(".JPG"):
        print image_directory+f_name

        # we want to map these new image back to the template image
        # this way we can use the template coordinates without having to do any further transformations
        # image = align(image_directory+template_image,image_directory+f_name)
        image = align(image_directory+f_name,image_directory+template_image)
        # cv2.imwrite("/home/ggdhines/t1.png",image)

        # # cv2.imwrite('/home/ggdhines/messigray.png',image)
        # # break
        # # print type(image)
        # # sub_image = image[162:238,274:326,:]
        r = range(162,238)
        c = range(274,326)
        sub_image = image[np.ix_(c, r)]

        img = Image.fromarray(sub_image, 'RGB')
        img.save('/home/ggdhines/a.png')


        extract(sub_image)
        raw_input("enter something")