__author__ = 'ggdhines'
from active_weather import ActiveWeather
import json
import matplotlib.pyplot as plt
import numpy as np
import Image
import math
import matplotlib.path as mplPath

class ActiveWeatherTesseract(ActiveWeather):
    def __init__(self):
        ActiveWeather.__init__(self)

    def __pixel_generator__(self,boundary):
        most_common_colour = [222,222,220]
        X,Y = zip(*boundary)
        x_min = int(math.ceil(min(X)))
        x_max = int(math.floor(max(X)))

        y_min = int(math.ceil(min(Y)))
        y_max = int(math.floor(max(Y)))


        cell = np.zeros((y_max-y_min+1,x_max-x_min+1,3),dtype=np.uint8)
        cell[:,:] = most_common_colour

        bbPath = mplPath.Path(np.asarray(boundary))

        for x in range(x_min,x_max+1):
            for y in range(y_min,y_max+1):

                if bbPath.contains_point((x,y)):
                    dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(self.image[y][x],most_common_colour)]))
                    if (dist > -1) and (self.template[y][x] != 0):
                        # plt.plot(x,y,"o",color="blue")
                        cell[y-y_min][x-x_min] = self.image[y][x]

        return cell

    def __process_subject__(self,f_name,region_id):
        self.__set_image__(f_name)

        cursor = self.conn.cursor()
        # c.execute("create table subject_info (subject_id int, fname text, template_id int)")
        cursor.execute("select template_id from subject_info where fname = \"" + f_name + "\"")
        r = cursor.fetchone()
        if r is None:
            assert self.template_id is not None
            cursor.execute("select count(*) from subject_info")
            subject_id = cursor.fetchone()[0]

            params = (subject_id,f_name,self.template_id)
            cursor.execute("insert into subject_info values(?,?,?)",params)
            self.conn.commit()

        cursor.execute("select template_id from subject_info where fname = \"" + f_name +"\"")
        template_id = cursor.fetchone()[0]

        cursor.execute("select column_filter,num_rows from templates where template_id = " + str(template_id) + " and region_id = " + str(region_id))
        columns_filter,num_rows = cursor.fetchone()
        columns_filter = json.loads(columns_filter)

        done = False

        #"create table cells (subject_id int, region_id int, column_id int, row_id int, pixels[][2], digit_index int, algorithm_classification int, probability float, gold_classification int)"
        cursor.execute("select column_id,row_id from cells where subject_id = " + str(self.subject_id) + " and region_id = " + str(region_id))
        previously_done = cursor.fetchall()

        for row_index in range(num_rows):
            if done:
                break
            for column_index in columns_filter:

                if (column_index,row_index) in previously_done:
                    continue

                cursor.execute("select boundary from cell_boundaries where template_id = " + str(template_id) + " and region_id = " + str(region_id) + " and column_id = " + str(column_index) + " and row_id = " + str(row_index))
                boundary_box = json.loads(cursor.fetchone()[0])
                pixels = self.__pixel_generator__(boundary_box)


                im = Image.fromarray(pixels,"RGB")
                im.save("/home/ggdhines/cells.jpeg")
                raw_input("enter something")




big_lower_x = 559
big_upper_x = 3245
big_lower_y = 1292
big_upper_y = 2014
record_region = (big_lower_x,big_upper_x,big_lower_y,big_upper_y)

with ActiveWeatherTesseract() as project:
    project.__set_template__("/home/ggdhines/t.png",record_region)
    # project.__plot__("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0241.JPG")
    # project.__set_columns__(0,0,[0,1,2,3,4,5,7,8,9,10,11,18,19])
    # project.__example_plot__("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0191.JPG",0,0,0)
    project.__process_subject__("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0191.JPG",0)