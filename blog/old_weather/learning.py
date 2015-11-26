# try:
#     import matplotlib
#     matplotlib.use('WXAgg')
# except ImportError:
#     pass

from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
try:
    import Image
except ImportError:
    from PIL import Image

from mnist import MNIST
from sklearn import neighbors
from sklearn.decomposition import PCA
from lines import __get_bounding_box__

class YGenerator:
    def __init__(self,x,ll,lr,ul,ur):
        ll_x,ll_y = ll
        lr_x,lr_y = lr
        ur_x,ur_y = ur
        ul_x,ul_y = ul

        # we can up to three lines which can act as lower bounds
        # the horizontal lower line (the obvious one) will always be a lower bound
        # but the vertical rhs and lhs lines can also be lower bounds if they are not completely
        # vertical - not sure how much this matters but probably good to protect against slightly rotated
        # cells which seem like a possibility

        lower_bounds = []

        # let's start with the horizontal lower line which goes from ll to lr
        # for the given x, what values of y does this line give?
        slope = (lr_y - ll_y)/float(lr_x-ll_x)
        y = ll_y + slope*(x-ll_x)
        lower_bounds.append(y)

        # next, let's look t the lhs - this is a lower bound if angling in slightly at the bottom
        # (probably draw an example to help understand)
        if ll_x > ul_x:
            slope = (ul_y - ll_y)/float(ul_x - ll_x)
            y = ll_y + slope*(x-ll_x)
            lower_bounds.append(y)

        # the rhs side is a lower bound if is angling in a but at the bottom:
        if lr_x < ur_x:
            slope = (ur_y - lr_y)/float(ur_x - lr_x)
            y = lr_y + slope*(x-lr_x)
            lower_bounds.append(y)

        # now take the maximum of all these values to be our starting point
        self.y_lower_bound = max(int(math.ceil(max(lower_bounds))),0)

        # now repeat for upper bound
        upper_bounds = []
        slope = (ur_y - ul_y)/float(ur_x-ul_x)
        y = ul_y + slope*(x-ul_x)
        upper_bounds.append(y)

        # look at the rhs and lhs lines - this time the inequalities are flipped (definitely draw a slanted
        # rectangle if that helps)
        if ll_x < ul_x:
            slope = (ul_y - ll_y)/float(ul_x - ll_x)
            y = ll_y + slope*(x-ll_x)
            upper_bounds.append(y)

        if lr_x > ur_x:
            slope = (ur_y - lr_y)/float(ur_x - lr_x)
            y = lr_y + slope*(x-lr_x)
            upper_bounds.append(y)


        self.y_upper_bound = int(math.floor(min(upper_bounds)))
        self.y = None


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.y is None:
            self.y = self.y_lower_bound
        else:
            self.y += 1

        if self.y == self.y_upper_bound:
            raise StopIteration()
        else:
            return self.y


class NearestNeighbours:
    def __init__(self,collect_gold_standard):
        n_neighbors = 15

        mndata = MNIST('/home/ggdhines/Databases/mnist')
        training = mndata.load_training()

        digits = range(0,10)

        training_dict = {d:[] for d in digits}

        for t_index,label in enumerate(training[1]):
            training_dict[label].append(training[0][t_index])

        weight = "distance"
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)

        pca = PCA(n_components=50)
        self.T = pca.fit(training[0])
        reduced_training = self.T.transform(training[0])
        # print sum(pca.explained_variance_ratio_)
        # clf.fit(training[0], training[1])
        self.clf.fit(reduced_training, training[1])

        self.transcribed_digits = {d:[] for d in digits}
        self.collect_gold_standard = collect_gold_standard

        self.cells_to_process = []
        self.completed_cells = []

    def __process_cell__(self,f_name,plot=False):
        image = Image.open(f_name)
        # lower_X,upper_X,lower_Y,upper_Y = self.__get_bounding_lines__(image)
        (lr_x,lr_y),(ur_x,ur_y),(ul_x,ul_y),(ll_x,ll_y) = __get_bounding_box__(image)

        # im = im.convert('L')#.convert('LA')
        image = np.asarray(image)

        colours = {}

        # the lhs will act as the lower bound for x
        # and we want the smallest integer x value greater than either ul_x,ll_x
        x_lower_bound = int(math.ceil(min(lr_x,ll_x)))
        x_upper_bound = int(math.floor(max(ur_x,ul_x)))

        for x in range(x_lower_bound,x_upper_bound):
            for y in YGenerator(x,(ll_x,ll_y),(lr_x,lr_y),(ul_x,ul_y),(ur_x,ur_y)):
                pixel_colour = tuple(image[y,x])
                # pixel_colour = int(image[r,c])
                # print pixel_colour
                if pixel_colour not in colours:
                    colours[pixel_colour] = 1
                else:
                    colours[pixel_colour] += 1

        most_common_colour,_ = sorted(colours.items(),key = lambda x:x[1],reverse=True)[0]
        pts = []


        # extract the ink pixels
        for x in range(x_lower_bound,x_upper_bound):
            for y in YGenerator(x,(ll_x,ll_y),(lr_x,lr_y),(ul_x,ul_y),(ur_x,ur_y)):
                pixel_colour = tuple(image[y,x])

                dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(pixel_colour,most_common_colour)]))

                # print dist

                if dist > 30:
                    if plot:
                        plt.plot(x,y,"o",color="blue")
                    pts.append((x,y))

        # return if we have an empty cell
        if pts == []:
            return

        if plot:
            plt.xlim((min(ll_x,ul_x),max(lr_x,ur_x)))
            plt.ylim((min(ur_y,ul_y),max(lr_y,ll_y)))
            plt.show()
            # return 0

        # convert to an numpy array because ... we have to
        pts = np.asarray(pts)
        gold_standard_digits,probabilities = self.__process_digits__(image,pts,most_common_colour)
        # print probabilities
        if probabilities != []:
            if min(probabilities) < 0.78:
                self.cells_to_process.append(f_name)
            else:
                self.completed_cells.append(f_name)

        if gold_standard_digits is None:
            return -1
        else:
            assert isinstance(gold_standard_digits,list)
            for pixels,digits in gold_standard_digits:
                self.transcribed_digits[digits].append(pixels)
            if probabilities == []:
                return None
            else:
                return min(probabilities)



    def __process_digits__(self,image,pts,most_common_colour):
        # do dbscan
        db = DBSCAN(eps=3, min_samples=20).fit(pts)
        labels = db.labels_
        unique_labels = set(labels)

        # each cluster should hopefully correspond to a different digit

        digit_probabilities = []
        gold_standard_digits = []

        colours = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        # print image.shape

        for k,col in zip(unique_labels,colours):
            # ignore noise
            if k == -1:
                continue

            # xy is the set of pixels in this cluster
            class_member_mask = (labels == k)
            xy = pts[class_member_mask]

            X_l,Y_l = zip(*xy)

            # we need to scale the digit so that it is the same size as the MNIST training examples
            # although the MNIST set is 28x28 pixels - there is a 4 pixel wide border around the digits
            # why? who knows. Anyways the actual height of the pixels that we want is 20
            max_x = max(X_l)
            max_y = max(Y_l)

            min_x = min(X_l)
            min_y = min(Y_l)
            # print k,(max_x,max_y)

            # plt.plot(X_l,Y_l,"o",color=col)

            #
            # plt.xlim((min_x+0,min_x+28))
            # plt.ylim((min_y+28,min_y+0))
            # plt.show()

            desired_height = 20.

            width_ratio = (max_x-min_x)/desired_height
            height_ratio = (max_y-min_y)/desired_height

            # calculate the resulting height or width - we want the maximum of these value to be 20
            if width_ratio > height_ratio:
                # wider than taller
                # todo - probably not a digit
                width = int(desired_height)
                height = int(desired_height*(max_y-min_y)/float(max_x-min_x))
            else:
                height = int(desired_height)
                # print (max_y-max_y)/float(max_x-min_x)
                width = int(desired_height*(max_x-min_x)/float(max_y-min_y))

            # the easiest way to do the rescaling is to make a subimage which is a box around the digit
            # and just get the Python library to do the rescaling - takes care of anti-aliasing for you :)
            # obviously this box could contain ink that isn't a part of this digit in particular
            # so we just need to be careful about what pixel we extract from the
            r = range(min_y,max_y+1)
            c = range(min_x,max_x+1)

            # print (min_y,max_y+1)
            # print (min_x,max_x+1)

            # todo - this will probably include noise-pixels, so we need to redo this
            template = image[np.ix_(r, c)]


            zero_template = np.zeros((len(r),len(c),3))


            for (x,y) in xy:
                # print (y-min_y,x-min_x),zero_template.shape
                # print zero_template[(y-min_y,x-min_x)]
                # print image[(y,x)]
                zero_template[(y-min_y,x-min_x)] = image[(y,x)]
            # cv2.imwrite("/home/ggdhines/aa.png",np.uint8(np.asarray(zero_template)))
            i = Image.fromarray(np.uint8(np.asarray(zero_template)))
            i.save("/home/ggdhines/aa.png")
            # assert False

            digit_image = Image.fromarray(np.uint8(np.asarray(zero_template)))
            # plt.show()
            # cv2.imwrite("/home/ggdhines/aa.png",np.uint8(np.asarray(template)))
            # raw_input("template extracted")
            # continue

            digit_image = digit_image.resize((width,height),Image.ANTIALIAS)

            # print zero_template.shape
            if min(digit_image.size) == 0:
                continue
            digit_image.save("/home/ggdhines/aa.png")
            # digit_image = digit_image.convert('L')

            grey_image =  np.asarray(digit_image.convert('L'))

            # # we need to center subject
            # if height == 28:
            #     # center width wise
            #
            #     y_offset = 0
            # else:
            #
            #     x_offset = 0

            x_offset = int(28/2 - width/2)
            y_offset = int(28/2 - height/2)

            digit_array = np.asarray(digit_image)

            centered_array = [0 for i in range(28**2)]

            # print digit_array == digit_image
            # print "===----"
            # try:
            #     darkest_pixel = 0
            #     for y in range(len(digit_array)):
            #         for x in range(len(digit_array[0])):
            #             darkest_pixel = max(darkest_pixel,digit_array[y][x])
            # except TypeError:
            #     print "problem skipping this one"
            #     continue

            # darkest_pixel = max(darkest_pixel,100)

            if self.collect_gold_standard:
                print (max_x-min_x),(max_y-min_y)
                print type(digit_array)
                print digit_array
            for y in range(len(digit_array)):
                for x in range(len(digit_array[0])):
                    # dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(digit_array[y][x],ref1)]))
                    # if dist1 > 10:
                    # if digit_array[y][x] > 0.4:
                    #     plt.plot(x+x_offset,y+y_offset,"o",color="blue")
                    # digit_array[y][x] = digit_array[y][x]/255.
                    # print digit_array[y][x] - most_common_colour
                    # dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(digit_array[y][x],most_common_colour)]))
                    dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(digit_array[y][x],(0,0,0))]))


                    if dist > 30:#digit_array[y][x] > 10:
                        centered_array[(y+y_offset)*28+(x+x_offset)] = grey_image[y][x]#/float(darkest_pixel)
                        if self.collect_gold_standard:
                            print "*",
                    else:
                        centered_array[(y+y_offset)*28+(x+x_offset)] = 0
                        if self.collect_gold_standard:
                            print " ",
                if self.collect_gold_standard:
                    print

            # print digit_probabilities
            if self.collect_gold_standard:
                digit = raw_input("enter digit - ")
                if digit == "":
                    return None
                elif digit == "u":
                    # if unknown, assume that the "digit" actually belongs to another cell
                    # so in practice the user won't transcribe it
                    pass
                else:
                    gold_standard_digits.append((centered_array,int(digit)))

            # for index,i in enumerate(centered_array):
            #     if i > 0:
            #         x = index%28
            #         y = index/28
            #         plt.plot(x,y,"o",color="blue")

            # plt.ylim((28,0))
            # plt.xlim((0,28))
            # plt.savefig("/home/ggdhines/tmp.png")
            # plt.close()


            centered_array = np.asarray(centered_array)
            # print centered_array
            centered_array = self.T.transform(centered_array)
            # print centered_array
            # print clf.predict_proba(centered_array)
            # raw_input("enter something")
            t = self.clf.predict_proba(centered_array)
            # print t
            # print list(t)
            # print t[0]
            # print
            # print t
            digit_probabilities.append(max(t[0]))
            # print t[0]



        max_y,max_x,_ = image.shape
        # print image.shape
        plt.xlim((0,max_x))
        plt.ylim((max_y,0))

        # if (len(unique_labels) > 0) and (unique_labels != {-1}):
        #     print digit_probabilities
        #     # plt.show()
        # else:
        #     plt.close()
        return gold_standard_digits,digit_probabilities

    def hesse_line_reduction(self,line_seg):
        """
        use if we want to cluster based on Hesse normal form - but want to retain the original values
        :param line_segment:
        :return:
        """
        (x1,y1),(x2,y2) = line_seg

        dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

        try:
            tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
            theta = math.atan(tan_theta)
        except ZeroDivisionError:
            theta = math.pi/2.

        return dist,theta

    def __normalize_lines__(self,intercepts,slopes):
        """
        normalize the lines so that the intercepts and slopes are all between 0 and 1
        makes cluster better
        also returns a dictionary which allows us to "unnormalize" lines so that we refer to the original values
        """
        mean_intercept = np.mean(intercepts)
        std_intercept = np.std(intercepts)

        normalized_intercepts = [(i-mean_intercept)/std_intercept for i in intercepts]

        mean_slopes = np.mean(slopes)
        std_slopes = np.std(slopes)

        normalized_slopes = [(s-mean_slopes)/std_slopes for s in slopes]

        return normalized_intercepts,normalized_slopes

    def __get_bounding_lines__(self,image):
        image = image.convert('L')
        image = np.asarray(image)

        edges = canny(image, 2, 1, 20)
        lines = probabilistic_hough_line(edges, threshold=2, line_length=5,line_gap=0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4), sharex=True, sharey=True)

        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_axis_off()
        ax1.set_adjustable('box-forced')

        ax2.imshow(edges, cmap=plt.cm.gray)
        ax2.set_title('Canny edges')
        ax2.set_axis_off()
        ax2.set_adjustable('box-forced')

        ax3.imshow(edges * 0)

        intercepts = []
        slopes = []

        for line in lines:
            p0, p1 = line
            dist,theta = self.hesse_line_reduction(line)
            intercepts.append(dist)
            slopes.append(theta)

        intercepts_n,slopes_n = self.__normalize_lines__(intercepts,slopes)
        normalized_lines = zip(intercepts_n,slopes_n)
        normalized_lines = np.asarray(normalized_lines)
        db = DBSCAN(eps=0.05, min_samples=1).fit(normalized_lines)

        labels = db.labels_
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        # print unique_labels
        # print labels

        height,width = image.shape

        lb_lines = []
        ub_lines = []
        rhs_lines = []
        lhs_lines = []
        horiz_lines = []

        threshold = 0.4

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # need to deal with these line segments independently
                continue
            else:
                in_cluster = [i for (i,l) in enumerate(labels) if l == k]

                # do we have a vertical or horiz. line?
                avg_slope = np.median([slopes[i] for i in in_cluster])
                segments = [lines[i] for i in in_cluster]

                # horiz.
                if (math.fabs(avg_slope) < 0.01):

                    # so the y. value is fixed, what are the y-value distances?
                    dist = sum([math.fabs(x2-x1) for ((x1,y1),(x2,y2)) in segments])
                    percent = dist/float(width)
                    avg_Y = np.median([(y1+y2)/2. for ((x1,y1),(x2,y2)) in segments])
                    # print "horiz.",percent,avg_Y < height/2.

                    if percent > threshold:


                        if avg_Y < height/2.:
                            lb_lines.append(segments)
                        else:
                            ub_lines.append(segments)


                        for ((x1,y1),(x2,y2)) in segments:
                            ax3.plot((x1,x2), (y1,y2),color=col)
                elif (math.fabs(avg_slope - math.pi/2.) < 0.01):
                    #vertical
                    # print "vertical"
                    dist = sum([math.fabs(y2-y1) for ((x1,y1),(x2,y2)) in segments])
                    percent = dist/float(height)
                    if percent > threshold:
                        avg_X = np.median([(x1+x2)/2. for ((x1,y1),(x2,y2)) in segments])

                        if avg_X < width/2.:
                            # print "****====="
                            lhs_lines.append(segments)
                        else:
                            rhs_lines.append(segments)


                        for ((x1,y1),(x2,y2)) in segments:
                            ax3.plot((x1,x2), (y1,y2),color=col)
                else:
                    # print "other"
                    # print math.fabs(avg_slope - math.pi/2.)
                    continue



            # if (math.fabs(theta) < 0.0001) or (math.fabs(theta - math.pi/2.) < 0.0001):
            #     ax3.plot((p0[0], p1[0]), (p0[1], p1[1]))

        # find the highest lower bound line - if there is one
        # this should be the inside of the line
        lower_Y = 0
        lower_X = 0
        upper_Y = height
        upper_X = width
        if lb_lines != []:
            for segments in lb_lines:
                print sorted(segments,key = lambda l:l[0][0])
            avg_Y = [np.median([(y1+y2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in lb_lines]
            lower_Y = max(avg_Y)

        print "****"

        if ub_lines != []:
            for segments in ub_lines:
                print sorted(segments,key = lambda l:l[0][0])
            avg_Y = [np.median([(y1+y2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in ub_lines]
            upper_Y = min(avg_Y)

        print "****"

        if rhs_lines != []:
            for segments in rhs_lines:
                print sorted(segments,key = lambda l:l[0][1])
            avg_X = [np.median([(x1+x2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in rhs_lines]
            upper_X = min(avg_X)

        print "****"

        if lhs_lines != []:
            for segments in lhs_lines:
                print sorted(segments,key = lambda l:l[0][1])
            avg_X = [np.median([(x1+x2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in lhs_lines]
            lower_X = max(avg_X)

        print "****"

        # print lower_X,upper_X
        # print lower_Y,upper_Y



        ax3.set_title('Probabilistic Hough')
        ax3.set_axis_off()
        ax3.set_adjustable('box-forced')
        plt.show()
        plt.close()

        # assert False

        return int(lower_X),int(upper_X),int(lower_Y),int(upper_Y)