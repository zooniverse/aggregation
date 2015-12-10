import matplotlib
matplotlib.use('WXAgg')
from skimage.transform import probabilistic_hough_line
from skimage.data import load
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
import matplotlib.path as mplPath
import sqlite3
import json
import os
import cPickle as pickle
import matplotlib.cbook as cbook
import random
from scipy.interpolate import interp1d
from numpy import trapz
from sklearn import metrics
import learning


def deter(a,b,c,d):
    """
    return the determinant of the 2x2 matrix (a,b),(c,d)
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    return a*d - c*b


def does_intersect(l1,l2):
    """
    do these two line segments intersect
    :param l1:
    :param l2:
    :return:
    """
    (x1,y1),(x2,y2) = l1
    (x3,y3),(x4,y4) = l2

    # do these line overlap on the x-axis?
    if ((x3 < x1) and (x4 >= x1)) or ((x3 >= x1) and (x3 <= x2)):

        # do these lines overlap on the y-axis?
        if ((y1 < y3) and (y2 >= y3)) or ((y1 >= y3) and (y1 <= y4)):
            return True
        return False
    return False


def intersection(h_line,v_line):
    """
    return the intersection of two line segments - one of which is vertical and the other is horizontal
    :param h_line:
    :param v_line:
    :return:
    """
    (x1,y1),(x2,y2) = h_line
    (x3,y3),(x4,y4) = v_line

    # do we have a completely vertical line
    try:
        if x3 == x4:
            intersect_x = x3

            m = (y2-y1)/float(x2-x1)
            b = y2 - m*x2
            intersect_y = m*intersect_x+b
        else:
            # see https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
            # for an explanation of the following math
            d1 = deter(x1,y1,x2,y2)
            d2 = deter(x1,1,x2,1)
            d3 = deter(x3,y3,x4,y4)
            d4 = deter(x3,1,x4,1)

            D1 = deter(d1,d2,d3,d4)

            d5 = deter(y1,1,y2,1)
            d6 = deter(y3,1,y4,1)

            D2 = deter(d1,d5,d3,d6)

            d7 = deter(x3,1,x4,1)
            d8 = deter(y3,1,y4,1)

            D3 = deter(d2,d5,d7,d8)

            intersect_x = D1/float(D3)
            intersect_y = D2/float(D3)
    except ZeroDivisionError:
        print h_line
        print v_line
        print D3
        raise

    return intersect_x,intersect_y

def multi_intersection(h_multi_line,v_multi_line):
    for h_index in range(len(h_multi_line)-1):
        h_line = h_multi_line[h_index:h_index+2]

        for v_index in range(len(v_multi_line)-1):
            v_line = v_multi_line[v_index:v_index+2]

            if does_intersect(h_line,v_line):
                try:
                    return h_index,v_index,intersection(h_line,v_line)
                except ZeroDivisionError:
                    print h_multi_line
                    print v_multi_line
                    raise
    assert False

def hesse_line(line_seg):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """


    (x1,y1),(x2,y2) = line_seg

    # x2 += random.uniform(-0.0001,0.0001)
    # x1 += random.uniform(-0.0001,0.0001)

    dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

    try:
        tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
        theta = math.atan(tan_theta)
    except ZeroDivisionError:
        theta = math.pi/2.

    return dist,theta

big_lower_x = 559
big_upper_x = 3245
big_lower_y = 1292
big_upper_y = 2014


# functions for sorting lines - has to deal with the fact that either lb_lines or ub_lines could be empty
def lowest_y((lb_lines,ub_lines)):
    if lb_lines == []:
        assert ub_lines != []
        return ub_lines[0][1]
    else:
        return lb_lines[0][1]


def lowest_x((lb_lines,ub_lines)):
    if lb_lines == []:
        assert ub_lines != []
        return ub_lines[0][0]
    else:
        return lb_lines[0][0]

def guass_kernel(x,x_star,b):
    return math.exp(-(x-x_star)**2/(2.*b**2))

class ActiveWeather:
    def __init__(self):
        self.template = None
        self.template_id = None
        self.h_lines = None
        self.v_lines = None

        self.image = None
        self.conn = None

        if os.path.isfile("/home/ggdhines/classifier.pickle"):
            self.classifier = pickle.load(open("/home/ggdhines/classifier.pickle","rb"))
        else:
            self.classifier = learning.NearestNeighbours()

        self.cluster = None



    def __db_setup__(self):
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        current_tables = [r[0] for r in c.fetchall()]


        if "subject_info" not in current_tables:
            c.execute("create table subject_info (subject_id int, fname text, template_id int)")

        if "templates" not in current_tables:
            c.execute("create table templates (template_id, fname text, region_id int, lower_x int, lower_y int, column_filter int[], num_rows int)")

        if "cell_boundaries" not in current_tables:
            c.execute("create table cell_boundaries (template_id int, region_id int, column_id int, row_id int, boundary int[][2])")

        if "cells" not in current_tables:
            print "redoing cells"
            c.execute("create table cells (subject_id int, region_id int, column_id int, row_id int, pixels[][2], digit_index int, algorithm_classification int, probability float, gold_classification int)")
            c.execute("CREATE UNIQUE INDEX t1b ON cells(subject_id,region_id,column_id,row_id,digit_index)")
        self.conn.commit()



    def __update_classifier__(self):
        self.classifier = blog.old_weather.learning.NearestNeighbours()

        cursor = self.conn.cursor()
        cursor.execute("select subject_id,region_id,column_id,row_id,digit_index,pixels from cells")

        for subject_id,region_id,column_id,row_id,digit_index,pixels in cursor.fetchall():
            cursor.execute("select fname from subject_info where subject_id = " + str(subject_id))
            fname = cursor.fetchone()[0]
            image = load(fname)

            pixels = json.loads(pixels)

            _,algorithm_digit,prob = self.classifier.__identify_digit__(image,pixels,collect_gold_standard=False)
            cursor.execute("update cells set algorithm_classification = " + str(algorithm_digit) + ", probability = " + str(prob) + " where subject_id = " + str(subject_id) + " and region_id = " + str(region_id) + " and column_id = " + str(column_id) + " and row_id = " + str(row_id) + " and digit_index = " + str(digit_index))
        self.conn.commit()

    def __accuracy__(self):
        cursor = self.conn.cursor()
        cursor.execute("select subject_id,region_id,column_id,row_id,algorithm_classification, probability, gold_classification from cells")

        probabilities = {}
        correctness = {}

        individual_total = 0
        individual_correct = 0.

        for (subject_id,region_id,column_id,row_id,alg,p,gold) in cursor.fetchall():
            # print alg,gold
            id_ = subject_id,region_id,column_id,row_id

            individual_total += 1

            if alg == gold:
                individual_correct += 1

            if id_ in probabilities:
                # probabilities[id_] = min(probabilities[id_],p)
                probabilities[id_] = probabilities[id_]*p
                correctness[id_] = correctness[id_] and (alg == gold)
            else:
                probabilities[id_] = p
                correctness[id_] = (alg == gold)

        print individual_correct/individual_total

        print sum([1 for c in correctness.values() if c])/float(len(correctness))

    def __set_image__(self,f_name):
        c = self.conn.cursor()
        c.execute("select subject_id from subject_info where fname = '" + str(f_name)+"'")
        r = c.fetchone()

        if r is None:
            c.execute("select count(*) from subject_info")
            self.subject_id = c.fetchone()[0]

            params = (self.subject_id,f_name,self.template_id)
            c.execute("insert into subject_info values(?,?,?)",params)
            self.conn.commit()
        else:
            self.subject_id = r[0]
        self.image = load(f_name)


    def __set_template__(self,fname,(big_lower_x,big_upper_x,big_lower_y,big_upper_y)):
        # have we used this template/region before?
        c = self.conn.cursor()
        c.execute("select lower_x,lower_y from templates where fname = '" + fname + "'")
        #"and lower_x = " + str(big_lower_x) + " and lower_y  = " + str(big_lower_y))

        self.template = load(fname)

        # so we have not used this template/region before
        if c.fetchall() == []:
            c.execute("select count(*) from templates")
            self.template_id = c.fetchone()[0]
            region_id = 0


            self.template_fname = fname
            self.template = load(self.template_fname)

            self.big_lower_x = big_lower_x
            self.big_upper_x = big_upper_x
            self.big_lower_y = big_lower_y
            self.big_upper_y = big_upper_y

            horiz_segments,vert_segments,horiz_intercepts,vert_intercepts = self.__get_grid_segments__()
            h_lines = self.__segments_to_grids__(horiz_segments,horiz_intercepts,horiz=True)
            v_lines = self.__segments_to_grids__(vert_segments,vert_intercepts,horiz=False)

            for row_index in range(len(h_lines)-1):
                for column_index in range(len(v_lines)-1):
                    boundary = self.__get_bounding_box__(h_lines,v_lines,row_index,column_index)
                    #(template_id int, region_id int, column_id int, row_id int, boundary int[][2])
                    params = (self.template_id,region_id,column_index,row_index,json.dumps(boundary))
                    c.execute("insert into cell_boundaries values (?,?,?,?,?)",params)

            params = (self.template_id,fname,region_id,big_lower_x,big_lower_y,json.dumps([]),row_index+1)
            c.execute("insert into templates values (?,?,?,?,?,?,?)",params)
        else:
            c.execute("select template_id from templates where fname = '" + fname + "'")
            self.template_id = c.fetchone()[0]

            self.big_lower_x = big_lower_x
            self.big_upper_x = big_upper_x
            self.big_lower_y = big_lower_y
            self.big_upper_y = big_upper_y

        self.conn.commit()

    def __plot__(self,fname):
        image_file = cbook.get_sample_data(fname)
        image = plt.imread(image_file)

        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(52,78)
        ax1.imshow(image)

        horiz_segments,vert_segments,horiz_intercepts,vert_intercepts = self.__get_grid_segments__()
        h_lines = self.__segments_to_grids__(horiz_segments,horiz_intercepts,horiz=True)
        v_lines = self.__segments_to_grids__(vert_segments,vert_intercepts,horiz=False)


        for (lb,ub) in h_lines:
            X,Y = zip(*lb)
            ax1.plot(X, Y,color="blue")
            X,Y = zip(*ub)
            ax1.plot(X, Y,color="blue")

        for (lb,ub) in v_lines:
            X,Y = zip(*lb)
            ax1.plot(X, Y,color="blue")
            X,Y = zip(*ub)
            ax1.plot(X, Y,color="blue")

        plt.savefig("/home/ggdhines/Databases/temp.jpg",bbox_inches='tight', pad_inches=0,dpi=72)

    def __get_grid_segments__(self):
        horiz_segments = []
        vert_segments = []

        horiz_intercepts = []
        vert_intercepts = []

        print "getting edges"
        edges = cv2.Canny(self.template,25,150,apertureSize = 3)

        print "probabilistic houghes"
        lines = probabilistic_hough_line(edges, threshold=5, line_length=3,line_gap=1)
        # plt.close()
        # fig, ax1 = plt.subplots(1, 1)
        # fig.set_size_inches(52,78)
        # ax1.imshow(self.image)


        for line in lines:
            p0, p1 = line
            X = p0[0],p1[0]
            Y = p0[1],p1[1]

            if (min(X) >= self.big_lower_x) and (max(X) <= self.big_upper_x) and (min(Y) >= self.big_lower_y) and (max(Y) <= self.big_upper_y):
                d,t = hesse_line(line)
                if math.fabs(t) <= 0.1:
                    # horiz_list.append(line)
                    # hesse_list.append(hesse_line(line))

                    m = (Y[0]-Y[1])/float(X[0]-X[1])
                    b = Y[0]-m*X[0]
                    horiz_intercepts.append(b+m*big_lower_x)
                    horiz_segments.append(line)
                elif math.fabs(t-math.pi/2.) <= 0.1:
                    # vert_list.append(line)
                    m = (X[0]-X[1])/float(Y[0]-Y[1])
                    b = X[0]-m*Y[0]
                    vert_intercepts.append(b+m*big_lower_y)
                    vert_segments.append(line)
                else:
                    continue

            # ax1.plot(X, Y,color="red")
        # plt.savefig("/home/ggdhines/Databases/new.jpg",bbox_inches='tight', pad_inches=0,dpi=72)
        return horiz_segments,vert_segments,horiz_intercepts,vert_intercepts

    def __pixel_generator__(self,boundary):
        most_common_colour = [222,222,220]
        X,Y = zip(*boundary)
        x_min = int(math.ceil(min(X)))
        x_max = int(math.floor(max(X)))

        y_min = int(math.ceil(min(Y)))
        y_max = int(math.floor(max(Y)))


        bbPath = mplPath.Path(np.asarray(boundary))

        ink_pixels = []

        for x in range(x_min,x_max+1):
            for y in range(y_min,y_max+1):

                if bbPath.contains_point((x,y)):
                    dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(self.image[y][x],most_common_colour)]))
                    if (dist > 40) and (self.template[y][x] != 0):
                        # plt.plot(x,y,"o",color="blue")
                        ink_pixels.append((x,y))

        # plt.xlim((x_min,x_max))
        # plt.ylim((y_max,y_min))
        # plt.show()
        # plt.close()

        return ink_pixels

    def __example_plot__(self,f_name,region_id,row_index,column_index):
        self.image = load(f_name)
        cursor = self.conn.cursor()

        cursor.execute("select template_id from subject_info where fname = \"" + f_name +"\"")
        template_id = cursor.fetchone()[0]

        cursor.execute("select boundary from cell_boundaries where template_id = " + str(template_id) + " and region_id = " + str(region_id) + " and column_id = " + str(column_index) + " and row_id = " + str(row_index))

        boundary_box = json.loads(cursor.fetchone()[0])

        x,y = zip(*boundary_box)
        x_max = int(max(x))
        y_max = int(max(y))
        x_min = int(min(x))
        y_min = int(min(y))


        sub_image = self.image[np.ix_(range(y_min,y_max+1), range(x_min,x_max+1))]
        fig, ax = plt.subplots()
        im = ax.imshow(sub_image)

        plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off')

        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

        plt.show()
        pixels = self.__pixel_generator__(boundary_box)
        self.__pixels_to_clusters__(pixels,True,y_min,y_max)



    def __segments_to_grids__(self,lines,intercepts,horiz=True):
        """
        cluster a whole bunch of different line segments
        :param lines:
        :param intercepts:
        :param horiz:
        :return:
        """
        retval = []


        X = np.asarray([[i,] for i in intercepts])
        # print X
        db = DBSCAN(eps=2, min_samples=1).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

                continue

            class_indices = [i for (i,l) in enumerate(labels) if l == k]
            class_size = sum([1 for (i,l) in enumerate(labels) if l == k])
            if class_size <= 3:
                continue

            multiline = []

            for i in class_indices:

                p0, p1 = lines[i]

                # if vertical - flip, then we'll flip back later
                if not horiz:
                    p0 = [p0[1],p0[0]]
                    p1 = [p1[1],p1[0]]

                # insert in increasing "X" order (quotation marks refer to the above flipping)
                # if horiz, sort by increasing X values

                if p0[0] < p1[0]:
                    multiline.append([list(p0),list(p1)])
                else:

                    multiline.append([list(p1),list(p0)])

            # sort so that we can add lines in order
            multiline.sort(key = lambda pt:pt[0][0])

            lb = set()
            ub = set()

            lb_lines = []
            ub_lines = []

            lower_distance = 0
            upper_distance = 0

            # note that these line segments will often not be overlapping but we want the ones that are since they will
            # they may provide upper and lower bounds on a line.
            for l_index,line in enumerate(multiline):
                for l2_index,line_2 in list(enumerate(multiline)):
                    if l_index == l2_index:
                        continue

                    #make sure that they are overlapping
                    (x1,y1),(x2,y2) = line
                    (x3,y3),(x4,y4) = line_2

                    # d1 = (x2-x1)
                    # d2 = (x4-x3)

                    # if line starts before line_2, make sure that it ends after line_2 has started
                    # or line starts before line_2 ends
                    # these two cases cover overlaps
                    if ((x1 < x3) and (x2 > x3)) or ((x1 >= x3) and (x1 <= x4)):
                        # we have an overlap
                        # the second case is slightly redundant but we have some strange cases
                        if (y1 < y3) and (y2 < y4):
                            lb.add(l_index)
                            ub.add(l2_index)

                            # lower_distance += d1
                            # upper_distance += d2
                        elif (y1 > y3) and (y2 > y4):
                            lb.add(l2_index)
                            ub.add(l_index)

                            # lower_distance += d2
                            # upper_distance += d1

            lb = sorted(list(lb))
            ub = sorted(list(ub))
            for i in lb:
                (x1,y1),(x2,y2) = multiline[i]
                if horiz:
                    # lb_lines.append(multiline[i])
                    lb_lines.extend(multiline[i])
                    d = x2 - x1
                else:
                    # else flip

                    lb_lines.extend(((y1,x1),(y2,x2)))
                    d = x2-x1
                lower_distance += d

            for i in ub:
                (x1,y1),(x2,y2) = multiline[i]
                if horiz:
                    ub_lines.extend(multiline[i])
                    d = x2 - x1
                else:
                    # else flip

                    ub_lines.extend([(y1,x1),(y2,x2)])
                    d = x2 - x1
                upper_distance += d



            if horiz:
                D = big_upper_x-big_lower_x
            else:
                D = big_upper_y-big_lower_y

            if lb_lines != []:
                X,Y = zip(*lb_lines)
                x_min = min(X)
                x_max = max(X)
                p1 = lower_distance/float(D)
            else:
                p1 = -1



            if ub_lines != []:
                X,Y = zip(*ub_lines)
                x_min = min(X)
                x_max = max(X)
                p2 = upper_distance/float(D)
            else:
                p2 = -1

            # if max(p1,p2) < 0.5:
            #     continue

            if True:
                if (lb_lines != []) and (ub_lines != []):
                    X_l,Y_l = zip(*lb_lines)
                    X_u,Y_u = zip(*ub_lines)

                    x_min = min(min(X_l),min(X_u))
                    x_max = max(max(X_l),max(X_u))

                    y_min = min(min(Y_l),min(Y_u))
                    y_max = max(max(Y_l),max(Y_u))
                else:
                    x_min = float("inf")
                    x_max = -float("inf")

                    y_min = float("inf")
                    y_max = -float("inf")

                for i,line in enumerate(multiline):
                    if (i not in lb) and (i not in ub):
                        (x1,y1),(x2,y2) = line

                        if horiz:
                            if (x2 < x_min) or (x1 > x_max):
                                lb_lines.extend([(x1,y1+1),(x2,y2+1)])
                                ub_lines.extend([(x1,y1-1),(x2,y2-1)])
                        else:
                            if (y2 < y_min) or (y1 > y_max):
                                lb_lines.extend([(y1,x1),(y2,x2)])
                                ub_lines.extend([(y1,x1),(y2,x2)])

            if horiz:
                lb_lines.sort(key = lambda pt:pt[0])
                ub_lines.sort(key = lambda pt:pt[0])
            else:
                lb_lines.sort(key = lambda pt:pt[1])
                ub_lines.sort(key = lambda pt:pt[1])
            # assert False
            if horiz:
                if lb_lines != []:
                    first_y = lb_lines[0][1]
                    last_y = lb_lines[-1][1]

                    lb_lines.insert(0,(big_lower_x,first_y))
                    lb_lines.append((big_upper_x,last_y))

                if ub_lines != []:
                    first_y = ub_lines[0][1]
                    last_y = ub_lines[-1][1]

                    ub_lines.insert(0,(big_lower_x,first_y))
                    ub_lines.append((big_upper_x,last_y))
            else:
                if lb_lines != []:
                    first_x = lb_lines[0][0]
                    last_x = lb_lines[-1][0]

                    lb_lines.insert(0,(first_x,big_lower_y))
                    lb_lines.append((last_x,big_upper_y))

                if ub_lines != []:
                    first_x = ub_lines[0][0]
                    last_x = ub_lines[-1][0]

                    ub_lines.insert(0,(first_x,big_lower_y))
                    ub_lines.append((last_x,big_upper_y))

            if True:
                if lb_lines != []:
                    X,Y = zip(*lb_lines)
                    # plt.plot(X,Y,"-",color="red")
                if ub_lines != []:
                    X,Y = zip(*ub_lines)
                    # plt.plot(X,Y,"-",color="blue")

                # for i,line in enumerate(multiline):
                #     if (i not in lb) and (i not in ub):
                #         if horiz:
                #             (x1,y1),(x2,y2) = line
                #         else:
                #             (y1,x1),(y2,x2) = line
                #
                #         plt.plot([x1,x2],[y1,y2],"-",color="green")
            if lb_lines != [] or ub_lines != []:
                # todo - what to do when they are both empty
                retval.append((lb_lines,ub_lines))

        if horiz:
            # assert isinstance(retval[0][0][0][1],int)
            retval.sort(key = lambda l:lowest_y(l))
        else:
            # assert isinstance(retval[0][0][0][0],int)
            retval.sort(key = lambda l:lowest_x(l))
        return retval

    def __get_bounding_box__(self,h_lines,v_lines,row_index,column_index):
        lower_horiz = h_lines[row_index][1]
        lower_vert = v_lines[column_index][1]

        x1_index,y1_index,(x1,y1) = multi_intersection(lower_horiz,lower_vert)
        # plt.plot(x1,y1,"o",color="yellow")

        upper_vert = v_lines[column_index+1][0]
        x2_index,y2_index,(x2,y2) = multi_intersection(lower_horiz,upper_vert)

        upper_horiz = h_lines[row_index+1][0]
        x3_index,y3_index,(x3,y3) = multi_intersection(upper_horiz,upper_vert)

        x4_index,y4_index,(x4,y4) = multi_intersection(upper_horiz,lower_vert)

        # let's draw out the bounding box/polygon
        offset = 0.01
        cell_boundries = [(x1+offset,y1+offset)]
        cell_boundries.extend([(x,y+offset) for (x,y) in lower_horiz[x1_index+1:x2_index+1]])
        cell_boundries.append((x2-offset,y2+offset))
        cell_boundries.extend([(x-offset,y) for (x,y) in upper_vert[y2_index+1:y3_index+1]])
        cell_boundries.append((x3-offset,y3-offset))
        cell_boundries.extend([(x,y-offset) for (x,y) in reversed(upper_horiz[x3_index+1:x4_index+1])])
        cell_boundries.append((x4+offset,y4-offset))
        cell_boundries.extend([(x+offset,y) for (x,y) in reversed(lower_vert[y1_index+1:y4_index+1])])
        cell_boundries.append((x1+offset,y1+offset))

        return cell_boundries

    def __pixels_to_clusters__(self,pixels,plot=False,y_max=None,y_min=None):
        pixels_np = np.asarray(pixels)
        db = DBSCAN(eps=3, min_samples=20).fit(pixels_np)
        labels = db.labels_
        unique_labels = set(labels)

        # each cluster should hopefully correspond to a different digit

        digit_probabilities = []
        gold_standard_digits = []

        colours = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        # print image.shape

        clusters = []

        for k,col in zip(unique_labels,colours):
            # ignore noise
            if k == -1:
                continue

            # xy is the set of pixels in this cluster
            class_member_mask = (labels == k)
            xy = pixels_np[class_member_mask]

            # X_l,Y_l = zip(*xy)
            if plot:
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=3)
            # digit,prob = self.__identify_digit__(xy)
            clusters.append(xy)

        x_values = [zip(*xy)[0] for xy in clusters]
        clusters_with_indices = list(enumerate(clusters))
        # sort in increasing x values
        clusters_with_indices.sort(key = lambda c_i:np.median(x_values[c_i[0]]))
        _,clusters = zip(*clusters_with_indices)
        if plot:
            plt.ylim((y_min,y_max))

            plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off')

            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')

            plt.show()
        return clusters

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

                if pixels == []:
                    continue

                clusters = self.__pixels_to_clusters__(pixels)
                if clusters == []:
                    continue
                print (column_index,row_index)

                for ii,c in enumerate(clusters):
                    # ("create table cells (subject_id int, region_id int, column_id int, row_id int, pixels[][2], digit_index int, algorithm_classification int, probability float, gold_classification int)")
                    try:
                        gold_standard,digit,digit_prob = self.classifier.__identify_digit__(self.image,c)
                    except ValueError:
                        done = True
                        break
                    params = (self.subject_id,region_id,column_index,row_index,json.dumps(c.tolist()),ii,digit,digit_prob,gold_standard)
                    # print (self.subject_id,region_id,column_index,row_index,json.dumps(c.tolist()),ii,digit,digit_prob,gold_standard)

                    cursor.execute("insert into cells values(?,?,?,?,?,?,?,?,?)",params)
                #

                if done:
                    break
                else:
                    self.conn.commit()

    def __enter__(self):
        self.conn = sqlite3.connect('/home/ggdhines/example.db')
        self.__db_setup__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # pickle.dump(self.classifier,open("/home/ggdhines/classifier.pickle","wb"))
        self.conn.close()

    def __set_columns__(self,template_id,region_id,columns):
        c = self.conn.cursor()
        c.execute("update templates set column_filter = \"" + json.dumps(columns) + "\" where template_id = " + str(template_id) + " and region_id = " + str(region_id))

    def __roc__(self):
        cursor = self.conn.cursor()
        cursor.execute("select subject_id,region_id,column_id,row_id,algorithm_classification, probability, gold_classification from cells")

        probabilities = {}
        correctness = {}

        for (subject_id,region_id,column_id,row_id,alg,p,gold) in cursor.fetchall():
            # print alg,gold
            id_ = subject_id,region_id,column_id,row_id
            if id_ in probabilities:
                # probabilities[id_] = min(probabilities[id_],p)
                probabilities[id_] = probabilities[id_]*p
                correctness[id_] = correctness[id_] and (alg == gold)
            else:
                probabilities[id_] = p
                correctness[id_] = (alg == gold)

        true_positive = []
        false_positive = []
        print len(probabilities)


        for key in correctness.keys():
            prob = probabilities[key]
            correct = correctness[key]

            if correct:
                true_positive.append(prob)

            else:
                false_positive.append(prob)


        roc_X = []
        roc_Y = []

        alpha_list = true_positive[:]
        alpha_list.extend(false_positive)
        alpha_list.sort()

        for alpha in alpha_list:
            positive_count = sum([1 for x in true_positive if x >= alpha])
            positive_rate = positive_count/float(len(true_positive))

            negative_count = sum([1 for x in false_positive if x >= alpha])
            negative_rate = negative_count/float(len(false_positive))

            roc_X.append(negative_rate)
            roc_Y.append(positive_rate)

        plt.plot(roc_X,roc_Y)

        plt.plot((0,1),(0,1),'--')
        plt.xlim((-0.01,1.00))
        plt.ylim((0,1.01))

        print trapz(list(reversed(roc_Y)),x=list(reversed(roc_X)))

        plt.show()

    def __probability_estimation__(self):
        cursor = self.conn.cursor()
        cursor.execute("select subject_id,region_id,column_id,row_id,algorithm_classification, probability, gold_classification from cells")

        probabilities = {}
        correctness = {}

        for (subject_id,region_id,column_id,row_id,alg,p,gold) in cursor.fetchall():
            # print alg,gold
            id_ = subject_id,region_id,column_id,row_id
            if id_ in probabilities:
                # probabilities[id_] = min(probabilities[id_],p)
                probabilities[id_] = probabilities[id_]*p
                correctness[id_] = correctness[id_] and (alg == gold)
            else:
                probabilities[id_] = p
                correctness[id_] = (alg == gold)

        training_indices = random.sample(range(len(probabilities)),25)
        testing_indices = [i for i in range(len(probabilities)) if i not in training_indices]

        xy_indices = [probabilities.keys()[i] for i in training_indices]
        x = np.asarray([probabilities[i] for i in xy_indices])
        y = np.asarray([correctness[i] for i in xy_indices])
        print x
        f = interp1d(x, y)
        f2 = interp1d(x, y, kind='cubic')
        xnew = np.linspace(min(x), max(x), num=41, endpoint=True)
        plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
        plt.ylim((0,1))
        plt.show()



    def __probability_smoothing__(self):
        cursor = self.conn.cursor()
        cursor.execute("select subject_id,region_id,column_id,row_id,algorithm_classification, probability, gold_classification from cells")

        probabilities = {}
        correctness = {}

        for (subject_id,region_id,column_id,row_id,alg,p,gold) in cursor.fetchall():
            # print alg,gold
            id_ = subject_id,region_id,column_id,row_id
            if id_ in probabilities:
                # probabilities[id_] = min(probabilities[id_],p)
                probabilities[id_] = probabilities[id_]*p
                correctness[id_] = correctness[id_] and (alg == gold)
            else:
                probabilities[id_] = p
                correctness[id_] = (alg == gold)

        temp = probabilities.keys()
        training_indices = random.sample(temp,25)
        testing_indices = [i for i in probabilities.keys() if i not in training_indices]

        # xy_indices = [[i] for i in training_indices]
        x_values = [probabilities[i] for i in training_indices]
        y_values = [correctness[i] for i in training_indices]

        y_est = []

        b = 0.15
        for x_star in np.arange(0,1,0.05):
            kernels = [guass_kernel(x,x_star,b) for x in x_values]
            y_hat = sum([k*y for (k,y) in zip(kernels,y_values)])/float(sum(kernels))
            y_est.append(y_hat)

        plt.plot(np.arange(0,1,0.05),y_est)
        plt.show()

        #####

        true_positive = []
        false_positive = []
        print len(probabilities)


        for key in testing_indices:
            prob = probabilities[key]
            kernels = [guass_kernel(x,prob,b) for x in x_values]
            y_values = [correctness[i] for i in training_indices]
            y_hat = sum([k*y for (k,y) in zip(kernels,y_values)])/float(sum(kernels))



            correct = correctness[key]

            if correct:
                true_positive.append(y_hat)

            else:
                false_positive.append(y_hat)


        roc_X = []
        roc_Y = []

        alpha_list = true_positive[:]
        alpha_list.extend(false_positive)
        alpha_list.sort()

        for alpha in alpha_list:
            positive_count = sum([1 for x in true_positive if x >= alpha])
            positive_rate = positive_count/float(len(true_positive))

            negative_count = sum([1 for x in false_positive if x >= alpha])
            negative_rate = negative_count/float(len(false_positive))

            roc_X.append(negative_rate)
            roc_Y.append(positive_rate)

        plt.plot(roc_X,roc_Y)

        print trapz(list(reversed(roc_Y)),x=list(reversed(roc_X)))

        plt.plot((0,1),(0,1),'--')
        plt.xlim((-0.01,1.00))
        plt.ylim((0,1.01))
        plt.show()

    def __smoothing_by_digit__(self):
        cursor = self.conn.cursor()
        cursor.execute("select subject_id,region_id,column_id,row_id,algorithm_classification, probability, gold_classification from cells")

        probabilities = {}
        correctness = {}

        predicted_digits = {i:[] for i in range(10)}

        ids = []

        predicted = []
        actual = []

        for (subject_id,region_id,column_id,row_id,alg,p,gold) in cursor.fetchall():
            # print alg,gold
            id_ = subject_id,region_id,column_id,row_id

            ids.append(id_)

        training_indices = random.sample(range(len(ids)),25)
        for ii in training_indices:
            subject_id,region_id,column_id,row_id = ids[ii]
            cursor.execute("select algorithm_classification, probability, gold_classification from cells where subject_id = " + str(subject_id) + " and region_id = " + str(region_id) + " and column_id = " + str(column_id) + " and row_id = " + str(row_id))
            for alg_digit,prob,gold_digit in cursor.fetchall():
                if alg_digit == -2:
                    continue

                predicted.append(alg_digit)
                actual.append(gold_digit)

                if alg_digit==gold_digit:
                    c_value = 1
                else:
                    c_value = 0

                predicted_digits[alg_digit].append((prob,c_value))

        testing_indices = [i for i in range(len(ids)) if i not in training_indices]

        b = 0.15

        true_positive = []
        false_positive = []

        for ii in testing_indices:
            subject_id,region_id,column_id,row_id = ids[ii]
            cursor.execute("select algorithm_classification, probability, gold_classification from cells where subject_id = " + str(subject_id) + " and region_id = " + str(region_id) + " and column_id = " + str(column_id) + " and row_id = " + str(row_id))

            overall_prob = 1.
            overall_correctness = True

            for alg_digit,prob,gold_digit in cursor.fetchall():
                if alg_digit == -2:
                    continue

                if predicted_digits[alg_digit] != []:
                    training_data = predicted_digits[alg_digit]
                    training_prob,training_correctness = zip(*training_data)

                    kernels = [guass_kernel(x,prob,b) for x in training_prob]
                    y_hat = sum([k*y for (k,y) in zip(kernels,training_correctness)])/float(sum(kernels))
                else:
                    y_hat = prob

                overall_prob = min(overall_prob,y_hat)
                overall_correctness = overall_correctness and (alg_digit == gold_digit)

            if overall_correctness:
                true_positive.append(overall_prob)
            else:
                false_positive.append(overall_prob)

        roc_X = []
        roc_Y = []

        alpha_list = true_positive[:]
        alpha_list.extend(false_positive)
        alpha_list.sort()

        plt.hist(alpha_list, bins=5, normed=1, histtype='step', cumulative=1)
        plt.ylim(0, 1.05)
        plt.show()

        for alpha in alpha_list:
            positive_count = sum([1 for x in true_positive if x >= alpha])
            positive_rate = positive_count/float(len(true_positive))

            negative_count = sum([1 for x in false_positive if x >= alpha])
            negative_rate = negative_count/float(len(false_positive))

            roc_X.append(1-negative_rate)
            roc_Y.append(positive_rate)

        plt.plot(roc_X,roc_Y)

        print trapz(list(reversed(roc_Y)),x=list(reversed(roc_X)))

        # plt.plot((0,1),(0,1),'--')
        plt.xlim((-0.01,1.00))
        plt.ylim((0,1.01))
        plt.xlabel("% of incorrect cells transcribed by users ")
        plt.ylabel("% of correct cells automatically transcribed ")
        plt.show()

        confusion_matrix = metrics.confusion_matrix(predicted,actual,labels= np.asarray([0,1,2,3,4,5,6,7,8,9,-1]))
        print confusion_matrix

record_region = (big_lower_x,big_upper_x,big_lower_y,big_upper_y)

with ActiveWeather() as project:
    project.__set_template__("/home/ggdhines/t.png",record_region)
    # project.__plot__("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0241.JPG")
    # project.__set_columns__(0,0,[0,1,2,3,4,5,7,8,9,10,11,18,19])
    # project.__example_plot__("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0191.JPG",0,0,0)
    # project.__process_subject__("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0191.JPG",0)

    # project.__update_classifier__()
    # project.__probability_smoothing__()
    # project.__smoothing_by_digit__()
    project.__accuracy__()