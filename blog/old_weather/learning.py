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

# class YGenerator:
#     def __init__(self,x,ll,lr,ul,ur):
#         ll_x,ll_y = ll
#         lr_x,lr_y = lr
#         ur_x,ur_y = ur
#         ul_x,ul_y = ul
#
#         # we can up to three lines which can act as lower bounds
#         # the horizontal lower line (the obvious one) will always be a lower bound
#         # but the vertical rhs and lhs lines can also be lower bounds if they are not completely
#         # vertical - not sure how much this matters but probably good to protect against slightly rotated
#         # cells which seem like a possibility
#
#         lower_bounds = []
#
#         # let's start with the horizontal lower line which goes from ll to lr
#         # for the given x, what values of y does this line give?
#         slope = (lr_y - ll_y)/float(lr_x-ll_x)
#         y = ll_y + slope*(x-ll_x)
#         lower_bounds.append(y)
#
#         # next, let's look t the lhs - this is a lower bound if angling in slightly at the bottom
#         # (probably draw an example to help understand)
#         if ll_x > ul_x:
#             slope = (ul_y - ll_y)/float(ul_x - ll_x)
#             y = ll_y + slope*(x-ll_x)
#             lower_bounds.append(y)
#
#         # the rhs side is a lower bound if is angling in a but at the bottom:
#         if lr_x < ur_x:
#             slope = (ur_y - lr_y)/float(ur_x - lr_x)
#             y = lr_y + slope*(x-lr_x)
#             lower_bounds.append(y)
#
#         # now take the maximum of all these values to be our starting point
#         self.y_lower_bound = max(int(math.ceil(max(lower_bounds))),0)
#
#         # now repeat for upper bound
#         upper_bounds = []
#         slope = (ur_y - ul_y)/float(ur_x-ul_x)
#         y = ul_y + slope*(x-ul_x)
#         upper_bounds.append(y)
#
#         # look at the rhs and lhs lines - this time the inequalities are flipped (definitely draw a slanted
#         # rectangle if that helps)
#         if ll_x < ul_x:
#             slope = (ul_y - ll_y)/float(ul_x - ll_x)
#             y = ll_y + slope*(x-ll_x)
#             upper_bounds.append(y)
#
#         if lr_x > ur_x:
#             slope = (ur_y - lr_y)/float(ur_x - lr_x)
#             y = lr_y + slope*(x-lr_x)
#             upper_bounds.append(y)
#
#
#         self.y_upper_bound = int(math.floor(min(upper_bounds)))
#         self.y = None
#
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         return self.next()
#
#     def next(self):
#         if self.y is None:
#             self.y = self.y_lower_bound
#         else:
#             self.y += 1
#
#         if self.y == self.y_upper_bound:
#             raise StopIteration()
#         else:
#             return self.y


class NearestNeighbours:
    def __init__(self):
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
        # self.collect_gold_standard = collect_gold_standard

        self.cells_to_process = []
        self.completed_cells = []


        # plt.show()

    def __set_image__(self,image):
        self.image = image

    def __identify_digit__(self,pts):
        """
        identify a cluster of pixels, given by pts
        image is needed for rescaling
        :param image:
        :param pts:
        :return:
        """
        gold_standard_digits = []
        digit_probabilities = []
        # do dbscan
        X,Y = zip(*pts)
        max_x = max(X)
        min_x = min(X)
        max_y = max(Y)
        min_y = min(Y)

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
        template = self.image[np.ix_(r, c)]


        zero_template = np.zeros((len(r),len(c),3))


        for (x,y) in pts:
            # print (y-min_y,x-min_x),zero_template.shape
            # print zero_template[(y-min_y,x-min_x)]
            # print image[(y,x)]
            zero_template[(y-min_y,x-min_x)] = self.image[(y,x)]
        # cv2.imwrite("/home/ggdhines/aa.png",np.uint8(np.asarray(zero_template)))
        # i = Image.fromarray(np.uint8(np.asarray(zero_template)))
        # i.save("/home/ggdhines/aa.png")
        # assert False

        digit_image = Image.fromarray(np.uint8(np.asarray(zero_template)))
        # plt.show()
        # cv2.imwrite("/home/ggdhines/aa.png",np.uint8(np.asarray(template)))
        # raw_input("template extracted")
        # continue

        digit_image = digit_image.resize((width,height),Image.ANTIALIAS)

        # print zero_template.shape
        if min(digit_image.size) == 0:
            return
        # digit_image.save("/home/ggdhines/aa.png")
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
                    print "*",
                else:
                    centered_array[(y+y_offset)*28+(x+x_offset)] = 0
                    print " ",
            print

        centered_array = np.asarray(centered_array)
        # print centered_array
        centered_array = self.T.transform(centered_array)
        # print centered_array
        # print clf.predict_proba(centered_array)
        # raw_input("enter something")
        t = self.clf.predict_proba(centered_array)
        digit_prob = max(t[0])
        digit_probabilities.append(max(t[0]))
        digit = list(t[0]).index(max(t[0]))
        print "knn thinks this is a " + str(digit) + " with probability " + str(max(t[0]))

        # print digit_probabilities

        digit = ""
        while digit == "":
            digit = raw_input("enter digit - ")
            if digit == "o":
                gold_standard = -1
            else:
                gold_standard = int(digit)



        return gold_standard,digit,digit_prob

