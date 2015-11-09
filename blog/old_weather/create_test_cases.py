import os
from template_align import align
import numpy as np
import Image
from extract_digits import extract

image_directory = "/home/ggdhines/Databases/old_weather/test_cases/"
template_image = "Bear-AG-29-1941-0414.JPG"


cell_columns = [(496,698),(698,805),(805,874),(1051,1234),(1405,1508),(1508,1719),(1719,1816),(1816,1927),(1927,2032),(2032,2134),(2733,2863),(2863,2971),(2971,3133)]
cell_rows = [(1267,1370),(1370,1428),(1428,1488),(1488,1547),(1547,1606),(1606,1665),(1665,1723),(1723,1781),(1781,1840),(1840,1899),(1899,1957),(1957,2016)]

for f_count,f_name in enumerate(os.listdir(image_directory)):
    f_name = "Bear-AG-29-1941-0557.JPG"
    if f_count == 1:
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

        for column_lb,column_ub in cell_columns:
            for row_lb,row_ub in cell_rows:
                r = range(row_lb,row_ub+1)
                c = range(column_lb,column_lb+1)

                sub_image = image[np.ix_(c, r)]

                img = Image.fromarray(sub_image, 'RGB')
                img.save('/home/ggdhines/a.png')


                extract(sub_image)
                raw_input("enter something")