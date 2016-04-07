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

from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import svm,metrics
import sqlite3

class Classifier:
    def __init__(self):
        self.conn = sqlite3.connect('/home/ggdhines/example.db')

    def __p_classification__(self,array):
        pass

    def __set_image__(self,image):
        self.image = image

    def __normalize_pixels__(self,image,pts):
        X,Y = zip(*pts)
        max_x = max(X)
        min_x = min(X)
        max_y = max(Y)
        min_y = min(Y)
        # print (max_x-min_x),(max_y-min_y)
        if (12 <= (max_x-min_x) <= 14) and (12 <= (max_y-min_y) <= 14):
            return -2,-2,1.

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


        for (x,y) in pts:
            # print (y-min_y,x-min_x),zero_template.shape
            # print zero_template[(y-min_y,x-min_x)]
            # print image[(y,x)]
            zero_template[(y-min_y,x-min_x)] = image[(y,x)]
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
                else:
                    centered_array[(y+y_offset)*28+(x+x_offset)] = 0

        return centered_array,28


    def __identify_digit__(self,image,pts,collect_gold_standard=True):
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

        centered_array,size = self.__normalize_pixels__(image,pts)

        algorithm_digit,digit_prob = self.__p_classification__(centered_array)



        # print digit_probabilities

        digit = ""
        if collect_gold_standard:
            for y in range(size):
                for x in range(size):
                    p = y*size+x
                    if centered_array[p] > 0:
                        print "*",
                    else:
                        print " ",
                print

            print "knn thinks this is a " + str(algorithm_digit) + " with probability " + str(digit_prob)
            while digit == "":
                digit = raw_input("enter digit - ")
                if digit == "o":
                    gold_standard = -1
                else:
                    gold_standard = int(digit)
        else:
            gold_standard = None


        return gold_standard,algorithm_digit,digit_prob


class NearestNeighbours(Classifier):
    def __init__(self):
        Classifier.__init__(self)

        n_neighbors = 25

        mndata = MNIST('/home/ggdhines/Databases/mnist')
        self.training = mndata.load_training()
        print type(self.training[0][0])

        weight = "distance"
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)

        pca = PCA(n_components=50)
        self.T = pca.fit(self.training[0])
        reduced_training = self.T.transform(self.training[0])
        print sum(pca.explained_variance_ratio_)
        # clf.fit(training[0], training[1])
        self.clf.fit(reduced_training, self.training[1])

    def __p_classification__(self,centered_array):
        centered_array = np.asarray(centered_array)
        # print centered_array
        centered_array = self.T.transform(centered_array)

        t = self.clf.predict_proba(centered_array)
        digit_prob = max(t[0])
        # digit_probabilities.append(max(t[0]))
        algorithm_digit = list(t[0]).index(max(t[0]))

        return algorithm_digit,digit_prob

class SVM(Classifier):
    def __init__(self):
        Classifier.__init__(self)

        self.classifier = svm.SVC(gamma=0.001,probability=True)

        mndata = MNIST('/home/ggdhines/Databases/mnist')
        training = mndata.load_training()

        self.classifier.fit(training[0], training[1])

class HierarchicalNN(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        cursor = self.conn.cursor()

        cursor.execute("select algorithm_classification, gold_classification from cells")
        r = cursor.fetchall()
        predicted,actual = zip(*r)

        confusion_matrix = metrics.confusion_matrix(predicted,actual,labels= np.asarray([0,1,2,3,4,5,6,7,8,9,-1,-2]))

        clusters = range(10)

        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                if confusion_matrix[i][j] > 3:
                    t = max(clusters[i],clusters[j])
                    t_old = min(clusters[i],clusters[j])
                    for k,c in enumerate(clusters):
                        if c == t_old:
                            clusters[k] = t

        print clusters

        mndata = MNIST('/home/ggdhines/Databases/mnist')
        training = mndata.load_training()

        testing = mndata.load_testing()

        labels = training[1]#[clusters[t] for t in training[1]]
        pca = PCA(n_components=50)
        self.T = pca.fit(training[0])

        f = self.T.components_.reshape((50,28,28))
        assert False

        reduced_training = self.T.transform(training[0])
        # starting variance explained
        print sum(pca.explained_variance_ratio_)

        # map classes
        mapped_labels = [clusters[i] for i in labels]

        weight = "distance"
        n_neighbors = 15
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
        self.clf.fit(reduced_training, labels)

        test_labels = testing[1]#[clusters[t] for t in testing[1]]
        reduced_testing = self.T.transform(testing[0])

        predictions = self.clf.predict(reduced_testing)

        print sum([1 for (t,p) in zip(test_labels,predictions) if t==p])/float(len(test_labels))

        # # filter
        # filtered_training = [(training[0][i],training[1][i]) for i in range(len(training[0])) if labels[training[1][i]] == 6]
        # data,new_labels = zip(*filtered_training)
        # self.T = pca.fit(data)
        # reduced_training = self.T.transform(data)
        # self.clf.fit(reduced_training, new_labels)
        #
        # filtered_testing = [(testing[0][i],testing[1][i]) for i in range(len(testing[0])) if labels[testing[1][i]] == 6]
        # testing,new_labels = zip(*filtered_testing)
        # reduced_testing = self.T.transform(testing)
        # predictions = self.clf.predict(reduced_testing)
        # print sum([1 for (t,p) in zip(new_labels,predictions) if t==p])/float(len(new_labels))