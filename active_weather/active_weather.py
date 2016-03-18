#!/usr/bin/env python
from __future__ import print_function
import matplotlib
# matplotlib.use('WXAgg')
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import database_connection
from scipy import stats
import tesseract_font
import cassandra
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
import random
# diabetes = datasets.load_diabetes()
# diabetes_X = diabetes.data[:, np.newaxis, 2]
# diabetes_X_train = diabetes_X[:-20]
# print(diabetes_X_train.shape)
# print(type(diabetes_X_train))
# assert False

class ActiveWeather:
    def __init__(self):
        try:
            self.cass_db = None
            self.cass_db = database_connection.Database()
            print("connected to the db")
        except cassandra.cluster.NoHostAvailable:
            print("could not connect to the db - will recalculate all values from scratch")
            self.cass_db = None

        # just for size reference
        # self.reference_subject = "Bear-AG-29-1940-0019"
        # self.reference_image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"+self.reference_subject+".JPG")
        # self.refer_shape = self.reference_image.shape
        #
        # self.horizontal_grid = self.cass_db.__get_horizontal_lines__(self.reference_subject,0)
        # self.vertical_grid = self.cass_db.__get_vertical_lines__(self.reference_subject,0)
        # # self.horizontal_grid,self.vertical_grid = self.__get_grid__()

        self.region = 0

        self.classifier = tesseract_font.ActiveTess()

    def __cross_validate__(self,pts):
        random.shuffle(pts)
        step = len(pts)/10

        x,y = zip(*pts)
        plt.plot(x,[-i for i in y],"o")
        plt.show()

        overall_err = []
        for i in range(10):
            validate_data = pts[i*step:(i+1)*step]
            test_data = pts[:i*step]
            test_data.extend(pts[(i+1)*step:])

            x,y = zip(*test_data)
            num_degrees = 5
            p = list(reversed(np.polyfit(x,y,num_degrees)))

            X,Y = zip(*validate_data)
            # predicted_y = [int(p[0]+p[1]*x+p[2]*x**2+p[3]*x**3) for x in X]

            predicted_y = [int(sum([p[d]*x**d for d in range(num_degrees+1)])) for x in X]

            err = sum([(p-y)**2 for (p,y) in zip(predicted_y,Y)])/float(len(Y))

            overall_err.append(err)

        print(np.mean(overall_err))





    def __interpolation__(self,shape,grid):
        template = np.zeros(shape,dtype=np.uint8)
        region_bounds = (559,3282,1276,2097)

        # use only the first 5 images - should be enough but we can change that if need be

        image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0019.JPG",0)
        horizontal_image = self.__sobel_image__(image,True)
        contours_to_return = self.__contour_extraction__(horizontal_image,True)

        # x,y = zip(*grid[0])
        cv2.drawContours(template,contours_to_return,-1,255,-1)

        zoomed_image = template[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]
        contours_to_return = self.__contour_extraction__(zoomed_image,True,False)

        all_x,all_y = [],[]

        new_template = np.zeros((zoomed_image.shape[0],zoomed_image.shape[1],3),dtype=np.uint8)
        for cnt in contours_to_return:
            perimeter = cv2.arcLength(cnt,True)

            if perimeter > 300:
                temp_template = np.zeros(zoomed_image.shape,dtype=np.uint8)
                cv2.drawContours(temp_template,[cnt],0,255,-1)
                plt.imshow(temp_template)
                plt.show()
                y,x = np.where(temp_template>0)

                self.__cross_validate__(zip(x,y))
                continue

                # temp_template = np.zeros(zoomed_image.shape,dtype=np.uint8)
                cv2.drawContours(new_template,[cnt],0,(255,255,255),-1)
                # x,y = np.where(template>0)
                # print(len(x))
                # plt.imshow(temp_template)
                # plt.show()

                # print(cnt)
                # cv2.drawContours(new_template,[cnt],0,255,-1)
                s = cnt.shape
                cnt = np.reshape(cnt,(s[0],s[2]))
                cnt = random.sample(cnt,1000)
                # x,y = zip(*cnt)
                new_x = sorted(list(set(x)))
                new_x = np.asarray(new_x)
                new_x = np.reshape(new_x,(new_x.shape[0],))

                print(min(x),max(x))

                x = np.asarray(x)
                x = np.reshape(x,(x.shape[0],))
                y = np.asarray(y)
                y = np.reshape(y,(y.shape[0],))

                # regr = linear_model.LinearRegression()
                # regr.fit(x, y)
                #
                # print("Residual sum of squares: %.2f"
                #   % np.median((regr.predict(x) - y) ** 2))
                #
                # print('Variance score: %.2f' % regr.score(x, y))
                # print((new_x[0],new_x[-1]))
                # print(len(new_x))

                p = np.polyfit(x,y,3)
                print(p)
                y_t = [int(p[3]+p[2]*x+p[1]*x**2+p[0]*x**3) for x in new_x]
                all_x.append(new_x)
                all_y.append(y_t)

                # for x in new_x:
                #     y = int(p[2]+p[1]*x+p[0]*x**2)
                #     print(x,y)
                    # new_template[y,x,:] = (255,0,0)
                # pts = np.asarray(zip(new_x,y_t))
                # print(pts.shape)
                # print(list(pts))
                # cv2.drawContours(new_template,[pts],0,(255,0,0),2)

                # cv2.drawContours(new_template,p,(255,0,0),1)
                # plt.plot(new_x,y_t,color='blue',linewidth=3)

        for x,y in zip(all_x,all_y):
            # print(x)
            # print(y)
            plt.plot(x,y,"-")

        plt.imshow(new_template)
        plt.show()


    def __directory_to_subjects__(self,directory):
        """
        take  directory of aligned images and convert them into column based subjects for upload to Panoptes
        :param directory:
        :return:
        """
        if directory[-1] != "/":
            directory += "/"

        # todo - make this more robust
        region_bounds = (559,3282,1276,2097)
        if self.cass_db is None:
            # we don't have a connection the db - so going to recalulate everything from scratch
            horizontal_grid,vertical_grid = self.__get_grid_for_table__(directory,region_bounds)
        else:
            # todo - read in from db
            horizontal_grid,vertical_grid = self.__get_grid_for_table__(directory,region_bounds)
            # self.horizontal_grid = self.cass_db.__get_horizontal_lines__(self.reference_subject,0)
            # self.vertical_grid = self.cass_db.__get_vertical_lines__(self.reference_subject,0)
            # todo - put this code inside the db call
            # uncomment - if you want to save the results to the cassandra db
            # self.cass_db.__add_horizontal_lines__(reference_subject,0,horizontal_lines)
            # self.cass_db.__add_vertical_lines__(reference_subject,0,vertical_lines)



        reference_image = cv2.imread("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0019.JPG")
        ref_shape = reference_image.shape[:2]

        # print(horizontal_grid)
        self.__interpolation__(ref_shape,horizontal_grid)
        assert False

        # todo - generalize to more than one region
        confidence_over_all_cells = []
        bad_count = 0
        for fname in glob.glob(directory+"*.JPG")[:15]:
            print(fname)
            # first_pass_columns, second_pass_columns, original_columns = self.__process_region__(fname,region_bounds,horizontal_grid,vertical_grid)

            files = self.__process_region__(fname,region_bounds,horizontal_grid,vertical_grid)

            for column_index,(init_threshold,pca_threshold) in enumerate(files):
                is_blank = self.classifier.__is_blank__(init_threshold)
                print(init_threshold)
                print(pca_threshold)
                if not is_blank:
                    text,column_confidence,bb_boxes = self.classifier.__process_column__(pca_threshold)
                    confidence_over_all_cells.extend(column_confidence)
                    print("==---")
                    print(text)
                    print(column_confidence)
                    text,column_confidence = self.__put_transcriptions_in_cells__(text,column_confidence,bb_boxes,horizontal_grid,vertical_grid,column_index,ref_shape)

                    for row_index,confidence in enumerate(column_confidence):
                        try:

                            if (confidence is not None) and (confidence < 80):
                                # print((bad_count,zip(text,column_confidence)))
                                border = self.__extract_cell_borders__(horizontal_grid,vertical_grid,row_index,column_index,ref_shape)
                                column = cv2.imread(pca_threshold)

                                # # todo - refactor and figure out why the heck column[border] doesn't work
                                for (x,y) in border:
                                    column[x,y,0] = 0
                                    column[x,y,1] = 0
                                    column[x,y,2] = 255

                                cv2.imwrite("/home/ggdhines/bad/bad_"+str(bad_count)+".jpg",column)

                                print((bad_count,confidence,text[row_index]))
                                bad_count += 1
                                raw_input("enter something")
                        except cv2.error:
                            print("error - skipping")

        # print(confidence_over_all_cells)
        n, bins, patches = plt.hist(confidence_over_all_cells, 80, normed=1,
                        histtype='step', cumulative=True)

        plt.show()

    def __put_transcriptions_in_cells__(self,text,confidence,bb_boxes,horizontal_grid,vertical_grid,column_index,reference_shape):
        column_confidence = []
        column_text = []

        for row_index in range(12):
            current_text = None
            current_confidence = None
            cell = self.__extract_cell_borders__(horizontal_grid,vertical_grid,row_index,column_index,reference_shape)
            cell_y,_ = zip(*cell)

            cell_top = min(cell_y)
            cell_bottom = max(cell_y)


            for index,(top,bottom) in enumerate(bb_boxes):

                in_cell = ((cell_bottom >= bottom) and (cell_top <= top)) or ((cell_bottom < bottom) and (cell_top > bottom))

                if in_cell and current_text is None:
                    current_text = text[index]
                    current_confidence = confidence[index]
                elif in_cell:
                    current_text += text[index]
                    current_confidence = min(current_confidence,confidence[index])

            column_confidence.append(current_confidence)
            column_text.append(current_text)

        return column_text,column_confidence

            # for column_index,(fname1,fname2,fname3) in enumerate(zip(first_pass_columns,second_pass_columns,original_columns)):
            #     is_blank = self.classifier.__is_blank__(fname1)
            #     if not is_blank:
            #         text,column_confidence = self.classifier.__process_column__(fname2)
            #         confidence_over_all_cells.extend(column_confidence)
            #
            #         for row_index,confidence in enumerate(column_confidence):
            #             if confidence < 50:
            #                 border = self.__extract_cell_borders__(horizontal_grid,vertical_grid,row_index,column_index,ref_shape)
            #                 column = cv2.imread(fname3)
            #
            #                 # # todo - refactor and figure out why the heck column[border] doesn't work
            #                 # for (x,y) in border:
            #                 #     column[x,y,0] = 0
            #                 #     column[x,y,1] = 0
            #                 #     column[x,y,2] = 255
            #
            #                 cv2.imwrite("/home/ggdhines/subject_"+str(bad_count)+".jpg",column)
            #                 shape = column.shape
            #                 s = column.reshape((shape[0]*shape[1],3)).astype(np.float)
            #                 pca = PCA(n_components=1)
            #                 print(s)
            #                 X_r = pca.fit_transform(s)
            #                 X_negative = X_r<0
            #                 X_r[X_negative] = 0
            #                 print(shape)
            #                 print(X_r)
            #                 print(X_r.shape)
            #                 redone_column = X_r.reshape(shape[:2])
            #                 plt.imshow(redone_column)
            #                 plt.show()
            #                 # # n, bins, patches = plt.hist(s, 50, normed=1, facecolor='green', alpha=0.5)
            #                 # m = np.median(s)
            #                 # column = column.astype(np.float)
            #                 # column -= m
            #                 # plt.imshow(column)
            #                 # plt.show()
            #                 #
            #                 #
            #                 # column = np.abs(column)
            #                 # background = column<60
            #                 # column[background] = 255
            #                 # cv2.normalize(column,column,0,255,cv2.NORM_MINMAX)
            #                 # cv2.imwrite("/home/ggdhines/testing.jpg",column)
            #                 # plt.imshow(column,cmap="gray")
            #                 # plt.show()
            #
            #
            #
            #
            #                 column = cv2.imread(fname2)
            #                 for (x,y) in border:
            #                     column[x,y,0] = 0
            #                     column[x,y,1] = 0
            #                     column[x,y,2] = 255
            #
            #                 cv2.imwrite("/home/ggdhines/thresholded_subject_"+str(bad_count)+".jpg",column)
            #
            #                 column = cv2.imread(fname1)
            #                 for (x,y) in border:
            #                     column[x,y,0] = 0
            #                     column[x,y,1] = 0
            #                     column[x,y,2] = 255
            #
            #                 cv2.imwrite("/home/ggdhines/thresholded1_subject_"+str(bad_count)+".jpg",column)
            #                 bad_count += 1
            #
            #                 text,column_confidence = self.classifier.__process_column__(fname1)
            #                 print(zip(text,column_confidence)[row_index])
            #                 text,column_confidence = self.classifier.__process_column__(fname2)
            #                 print(zip(text,column_confidence)[row_index])
            #                 assert False



    def __extract_cell_borders__(self,horizontal_grid,vertical_grid,row_index,column_index,reference_shape,fname=None):
        """
        :param image:
        :param v_index:
        :return:
        """

        mask = np.zeros(reference_shape,np.uint8)
        mask2 = np.zeros(reference_shape,np.uint8)
        cv2.drawContours(mask,horizontal_grid,row_index,255,-1)
        cv2.drawContours(mask,horizontal_grid,row_index+1,255,-1)
        cv2.drawContours(mask,vertical_grid,column_index,255,-1)
        cv2.drawContours(mask,vertical_grid,column_index+1,255,-1)

        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        # looking for the one interior contour
        for c,h in zip(contours,hier[0]):
            if h[-1] == -1:
                continue

            cv2.drawContours(mask2,[c],0,255,1)

        border_y,border_x = np.where(mask2>0)


        # now we need to normalize these values - relative to the region we are extracting them from
        t = horizontal_grid[0]
        _,min_y = np.min(t,axis=0)

        border_y -= min_y

        # now make the x values relative to the column we are extracting them from
        t = vertical_grid[column_index]
        min_x,_ = np.min(t,axis=0)

        border_x -= min_x

        return zip(border_y,border_x)

    def __sobel_image__(self,image,horizontal):
        """
        apply the sobel operator to a given image on either the vertical or horizontal axis
        basically copied from
        http://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
        :param horizontal:
        :return:
        """
        if horizontal:
            dy = cv2.Sobel(image,cv2.CV_16S,0,2)
            dy = cv2.convertScaleAbs(dy)
            cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        else:
            dx = cv2.Sobel(image,cv2.CV_16S,2,0)
            dx = cv2.convertScaleAbs(dx)
            cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
            ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernel)

        return close

    def __contour_extraction__(self,image,horizontal,approximation = True):
        """
        extract all the horizontal or vertical contours from an image
        strongly inspired by
        http://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
        :param image:
        :param horizontal:
        :return:
        """
        contours_to_return = []
        if approximation:
            _,contour, hier = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        else:
            _,contour, hier = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if (horizontal and w/h > 5) or ((not horizontal) and h/w > 5):
                contours_to_return.append(cnt)

        return contours_to_return

    def __get_contour_lines_over_image__(self,directory,horizontal,fname):
        """
        return the contours lines for a subject set of already aligned subjects
        if horizontal, return only the horizontal contours. Otherwise, return the vertical contours
        returns the contours over all the image - we still have to trim to the specific region
        :param horizontal:
        :return:
        """
        # todo - currently hard coded to work with only Bear 1940
        # lined_images is the set of every aligned image after we have applied the sobel operator to it
        # i.e. extracted either the vertical or horizontal lines
        lined_images = []

        # use only the first 5 images - should be enough but we can change that if need be
        files = sorted(list(glob.glob(directory+"*.JPG")))
        index = files.index(fname)

        for f in files[index:index+5]:
            image = cv2.imread(f,0)
            lined_images.append(self.__sobel_image__(image,horizontal))

        # the average image is the 40th percentile
        average_image = np.percentile(lined_images,40,axis=0)
        # convert back to np.uint8 so we have a proper image
        average_image = average_image.astype(np.uint8)

        if horizontal:
            cv2.imwrite("/home/ggdhines/horizontal_average.jpg",average_image)
        else:
            cv2.imwrite("/home/ggdhines/vertical_image.jpg",average_image)

        contours_to_return = self.__contour_extraction__(average_image,horizontal)

        return contours_to_return

    def __get_grid_for_table__(self,directory,region,fname):
        """
        directory - contains a set of aligned images
        extract the grid for a given region/table
        the region/table is specified by min_x,max_x,min_y,max_y
        :return:
        """
        assert region[0]<region[1]
        assert region[2]<region[3]
        # todo - refactor!!
        horizontal_lines = []
        vertical_lines = []
        # extract all horizontal lines
        horizontal_contours = self.__get_contour_lines_over_image__(directory,True,fname)

        # useful for when you want to draw out the image - just for debugging
        mask = np.zeros((3744,5616),dtype=np.uint8)
        delta = 50

        for cnt in horizontal_contours:
            shape = cnt.shape
            cnt = cnt.reshape((shape[0],shape[2]))
            max_x,max_y = np.max(cnt,axis=0)
            min_x,min_y = np.min(cnt,axis=0)

            if (min_y>=region[2]-delta) and (max_y<=region[3]+delta):
                # sanity check - if this an actual grid line - or just a blip?
                perimeter = cv2.arcLength(cnt,True)

                if perimeter > 100:
                    horizontal_lines.append(cnt)
                    # cv2.drawContours(mask,[cnt],0,255,1)

        horizontal_lines.sort(key = lambda l:l[0][1])

        vertical_contours = self.__get_contour_lines_over_image__(directory,False,fname)

        delta = 400
        for cnt in vertical_contours:
            shape = cnt.shape
            cnt = cnt.reshape((shape[0],shape[2]))
            max_x,max_y = np.max(cnt,axis=0)
            min_x,min_y = np.min(cnt,axis=0)

            interior_line = (min_x >= region[0]-100) and (max_x <= region[1]+100)and(min_y>=region[2]-delta) and (max_y<=region[3]+delta)
            through_line = (min_x >= region[0]-100) and (max_x <= region[1]+100) and (min_y < region[2]) and(max_y > region[3])

            if interior_line or through_line:

                perimeter = cv2.arcLength(cnt,True)
                if perimeter > 1000:

                    vertical_lines.append(cnt)

        vertical_lines.sort(key = lambda l:l[0][0])
        cv2.drawContours(mask,vertical_lines,0,255,-1)
        cv2.drawContours(mask,vertical_lines,1,255,-1)
        return horizontal_lines,vertical_lines

    def __extract_column__(self,image,column_index,vertical_grid,region_bounds):
        # get the region coordinates - so we can convert global grid line coordinates to
        # local ones (relative to just the grid line)

        t = vertical_grid[column_index]
        # t = t.reshape((t.shape[0],t.shape[2]))
        min_x,_ = np.min(t,axis=0)
        t = vertical_grid[column_index+1]
        # t = t.reshape((t.shape[0],t.shape[2]))
        max_x,_ = np.max(t,axis=0)

        # print(((min_x,max_x,region_bounds[0])))

        column = image[:,(min_x-region_bounds[0]):(max_x-region_bounds[0]+1)]

        return column

    def __region_mask__(self,reference_image,horizontal_grid,vertical_grid):
        """
        use the first and last horizontal/vertical grid lines to make a mask around the desired region/table
        :return:
        """
        reference_shape = reference_image.shape
        # [:2] in case we read in the image in colour format - doesn't seem necessary to throw an error
        # the first mask will be an outline of the region, sort of like #. The second mask will fill in the
        # central interior box
        mask = np.zeros(reference_shape[:2],np.uint8)
        mask2 = np.zeros(mask.shape,np.uint8)
        # draw the first and last horizontal/vertical grid lines to create a box
        cv2.drawContours(mask,horizontal_grid,0,255,-1)
        cv2.drawContours(mask,horizontal_grid,len(horizontal_grid)-2,255,-1)
        cv2.drawContours(mask,vertical_grid,0,255,-1)
        cv2.drawContours(mask,vertical_grid,len(vertical_grid)-1,255,-1)

        # find the (hopefully) one interior contour - should be our mask
        _,contours, hier = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        assert len(contours) == 1
        for c,h in zip(contours,hier[0]):
            if h[-1] == -1:
                continue

            cv2.drawContours(mask2,[c],0,255,-1)

        return mask2

    def __process_region__(self,fname,region_bounds,horizontal_grid,vertical_grid):
        files = []
        initial_threshed_image = self.__initial_threshold__(fname,region_bounds,horizontal_grid,vertical_grid)
        threshed_image = self.__pca_thresholding__(fname,region_bounds,horizontal_grid,vertical_grid)

        for column_index in range(len(vertical_grid)-1):
            init_column = self.__extract_column__(initial_threshed_image,column_index,vertical_grid,region_bounds)
            fname1 = "/home/ggdhines/active/init_"+str(column_index)+".jpg"
            cv2.imwrite(fname1,init_column)

            column = self.__extract_column__(threshed_image,column_index,vertical_grid,region_bounds)
            fname = "/home/ggdhines/active/pca_"+str(column_index)+".jpg"
            cv2.imwrite(fname,column)
            files.append((fname1,fname))

        return files

        # assert False
        # first_files = []
        # second_files = []
        # original_files= []
        # first_pass,second_pass = self.__extract_region__(fname,region_bounds,horizontal_grid,vertical_grid)
        #
        # # first
        # for column_index in range(len(vertical_grid)-1):
        #     column = self.__extract_column__(first_pass,column_index,vertical_grid,region_bounds)
        #     fname = "/home/ggdhines/first_"+str(column_index)+".jpg"
        #     cv2.imwrite(fname,column)
        #     first_files.append(fname)
        #
        # # first
        # for column_index in range(len(vertical_grid)-1):
        #     column = self.__extract_column__(second_pass,column_index,vertical_grid,region_bounds)
        #     fname = "/home/ggdhines/second_"+str(column_index)+".jpg"
        #     cv2.imwrite(fname,column)
        #     second_files.append(fname)
        #
        # for column_index in range(len(vertical_grid)-1):
        #     column = self.__extract_column__(original,column_index,vertical_grid,region_bounds)
        #     fname = "/home/ggdhines/original_"+str(column_index)+".jpg"
        #     cv2.imwrite(fname,column)
        #     original_files.append(fname)
        #
        # return first_files,second_files,original_files

    def __pca_thresholding__(self,fname,region_bounds,horizontal_grid,vertical_grid):
        # image = cv2.imread(fname,0)
        # second_pass = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,301,2)
        # cv2.drawContours(second_pass,horizontal_grid,-1,255,-1)
        # cv2.drawContours(second_pass,vertical_grid,-1,255,-1)
        # second_pass = self.__image_clean__(second_pass)
        # # zoom in
        # second_pass = second_pass[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]
        # plt.imshow(second_pass,cmap="gray")
        # plt.show()

        image = cv2.imread(fname)
        image_shape = image.shape
        # cv2.drawContours(image,horizontal_grid,-1,255,-1)
        # cv2.drawContours(image,vertical_grid,-1,255,-1)
        original = image[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]

        gray_image = cv2.imread(fname,0)
        zoomed_gray = gray_image[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]

        # temp = cv2.imread(fname,0)
        # temp = temp[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]
        # plt.imshow(temp)
        # plt.show()

        s = original.shape
        flat_image = original.reshape((s[0]*s[1],3))
        pca = PCA(n_components=1)

        X_r = pca.fit_transform(flat_image)
        print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))
        X_negative = X_r<0
        X_r[X_negative] = 0
        image = X_r.reshape((s[0],s[1]))

        template = np.zeros(image_shape[:2],np.uint8)
        template[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1] = image
        cv2.drawContours(template,horizontal_grid,-1,0,-1)
        cv2.drawContours(template,vertical_grid,-1,0,-1)
        zoomed_image = template[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]

        y_pts,x_pts = np.where(zoomed_image>0)
        X = np.asarray(zip(x_pts,y_pts))
        print("doing dbscan: " + str(X.shape))
        db = DBSCAN(eps=1, min_samples=5).fit(X)

        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        return_image = np.zeros(zoomed_image.shape,np.uint8)
        return_image.fill(255)

        print("going through dbscan results")
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                continue

            class_member_mask = (labels == k)
            # temp = np.zeros(X.shape)

            xy = X[class_member_mask]

            max_value = zoomed_image[xy[:, 1], xy[:, 0]].max()

            if max_value >= 110:
                # print(max_value)
                # plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=col,
                #          markeredgecolor='k')
                return_image[xy[:, 1], xy[:, 0]] = zoomed_gray[xy[:, 1], xy[:, 0]]

        print(np.max(return_image))
        cv2.imwrite("/home/ggdhines/testing.jpg",return_image)
        raw_input("hell world")

        return return_image

    def __initial_threshold__(self,fname,region_bounds,horizontal_grid,vertical_grid):
        """
        open fname, "zoom in" on the desired region, apply thresholding to "clean it up"
        region_bounds = min_x,max_x,min_y,max_y
        :param fname:
        :param region:
        :param mask:
        :return:
        """
        image = cv2.imread(fname,0)

        # uncomment if you want to apply ostu thresholding
        # see http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0
        _,first_pass = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.drawContours(first_pass,horizontal_grid,-1,255,-1)
        cv2.drawContours(first_pass,vertical_grid,-1,255,-1)
        first_pass = self.__image_clean__(first_pass)
        # zoom in
        first_pass = first_pass[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]

        # second_pass = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,301,2)
        # cv2.drawContours(second_pass,horizontal_grid,-1,255,-1)
        # cv2.drawContours(second_pass,vertical_grid,-1,255,-1)
        # second_pass = self.__image_clean__(second_pass)
        # # zoom in
        # second_pass = second_pass[region_bounds[2]:region_bounds[3]+1,region_bounds[0]:region_bounds[1]+1]

        return first_pass


    def __image_clean__(self,image):
        """
        after removing grid lines and applying thresholding, we will probably still have small "ticks" - bits of the
        grid line which weren't removed but can still cause problems for Tesseract (and probably other approaches too)
        """
        _,contours, hier = cv2.findContours(image.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # contours are probably in sorted order but just to be sure
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            if (h <= 7) or (w <= 7) or (perimeter <= 30):
                cv2.drawContours(image,[cnt],0,255,-1)

        return image

    def __read_box__(self):
        image = cv2.imread("/home/ggdhines/step4.jpg")
        s = image.shape
        with open("/home/ggdhines/boxout","r") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            for row in spamreader:
                _,x1,y1,x2,y2,_ = row
                cv2.rectangle(image,(int(x1),s[0]-int(y1)),(int(x2),s[0]-int(y2)),(255,0,0),2)

        image = cv2.imwrite("/home/ggdhines/step5.jpg",image)

if __name__ == "__main__":
    project = ActiveWeather()
    project.__directory_to_subjects__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/")
    # project.__image_threshold__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__extract_table__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__extract_column__(8)
    # project.__process_image__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # for i in range(10):
    #     project.__process_row__(i)
    # project.__process_image__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")
    # project.__remove_template__("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0720.JPG")\
    # project.__read_box__()