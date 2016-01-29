import os
# import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# import math
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
# from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# from scipy import spatial
from sklearn.cluster import DBSCAN
# import Image
# import cv2
from mnist import MNIST
from sklearn import neighbors
from sklearn.decomposition import PCA

n_neighbors = 15

mndata = MNIST('/home/ggdhines/Databases/mnist')
training = mndata.load_training()

weight = "distance"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)

pca = PCA(n_components=50)
T = pca.fit(training[0])
reduced_training = T.transform(training[0])
print sum(pca.explained_variance_ratio_)
# clf.fit(training[0], training[1])
clf.fit(reduced_training, training[1])

def hesse_line_reduction(line_seg):
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

image_directory = "/home/ggdhines/Databases/old_weather/cells/"

# cell_columns = [(496,698),(698,805),(805,874),(1051,1234),(1405,1508),(1508,1719),(1719,1816),(1816,1927),(1927,2032),(2032,2134),(2733,2863),(2863,2971),(2971,3133)]
# cell_rows = [(1267,1370),(1370,1428),(1428,1488),(1488,1547),(1547,1606),(1606,1665),(1665,1723),(1723,1781),(1781,1840),(1840,1899),(1899,1957),(1957,2016)]

cell_columns = [(510,713),(713,821),(821,890),(1067,1252),(1527,1739),(1739,1837),(1837,1949),(1949,2053),(2053,2156)]
cell_rows = [(1226,1320),(1320,1377)]

# s3://zooniverse-static/old-weather-2015/War_in_the_Arctic/Greenland_Patrol/Navy/Bear_AG-29_/

log_pages = list(os.listdir(image_directory))


def __normalize_lines__(intercepts,slopes):
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




def __get_bounding_lines__(image):
    image = image.convert('L')
    image = np.asarray(image)

    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=10,
                                     line_gap=0)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4), sharex=True, sharey=True)
    #
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    # ax1.set_axis_off()
    # ax1.set_adjustable('box-forced')
    #
    # ax2.imshow(edges, cmap=plt.cm.gray)
    # ax2.set_title('Canny edges')
    # ax2.set_axis_off()
    # ax2.set_adjustable('box-forced')
    #
    # ax3.imshow(edges * 0)

    intercepts = []
    slopes = []

    for line in lines:
        p0, p1 = line
        dist,theta = hesse_line_reduction(line)
        intercepts.append(dist)
        slopes.append(theta)

    intercepts_n,slopes_n = __normalize_lines__(intercepts,slopes)
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
            if (math.fabs(avg_slope) < 0.0001):
                print "horiz."
                # so the y. value is fixed, what are the y-value distances?
                dist = sum([math.fabs(x2-x1) for ((x1,y1),(x2,y2)) in segments])
                percent = dist/float(width)

                if percent > 0.5:
                    avg_Y = np.median([(y1+y2)/2. for ((x1,y1),(x2,y2)) in segments])

                    if avg_Y < height/2.:
                        lb_lines.append(segments)
                    else:
                        ub_lines.append(segments)

                pass
            elif (math.fabs(avg_slope - math.pi/2.) < 0.0001):
                #vertical
                print "vertical"
                dist = sum([math.fabs(y2-y1) for ((x1,y1),(x2,y2)) in segments])
                percent = dist/float(height)
                if percent > 0.5:
                    avg_X = np.median([(x1+x2)/2. for ((x1,y1),(x2,y2)) in segments])

                    if avg_X < width/2.:
                        lhs_lines.append(segments)
                    else:
                        rhs_lines.append(segments)

                pass
            else:
                print "other"
                continue

            # if percent > 0.5:
            #     for ((x1,y1),(x2,y2)) in segments:
            #         ax3.plot((x1,x2), (y1,y2),color=col)

        # if (math.fabs(theta) < 0.0001) or (math.fabs(theta - math.pi/2.) < 0.0001):
        #     ax3.plot((p0[0], p1[0]), (p0[1], p1[1]))

    # find the highest lower bound line - if there is one
    # this should be the inside of the line
    lower_Y = 0
    lower_X = 0
    upper_Y = height
    upper_X = width
    if lb_lines != []:
        avg_Y = [np.median([(y1+y2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in lb_lines]
        lower_Y = max(avg_Y)

    if ub_lines != []:
        avg_Y = [np.median([(y1+y2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in ub_lines]
        upper_Y = min(avg_Y)

    if rhs_lines != []:
        avg_X = [np.median([(x1+x2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in rhs_lines]
        upper_X = min(avg_X)

    if lhs_lines != []:
        avg_X = [np.median([(x1+x2)/2. for ((x1,y1),(x2,y2)) in segments]) for segments in lhs_lines]
        lower_X = max(avg_X)

    print lower_X,upper_X
    print lower_Y,upper_Y

    return int(lower_X),int(upper_X),int(lower_Y),int(upper_Y)

    # ax3.set_title('Probabilistic Hough')
    # ax3.set_axis_off()
    # ax3.set_adjustable('box-forced')
    # plt.show()

def __extract__(image):
    lower_X,upper_X,lower_Y,upper_Y = __get_bounding_lines__(image)

    # im = im.convert('L')#.convert('LA')
    image = np.asarray(image)

    colours = {}

    # under the assumption that most of the cell is not ink - find the most common pixel colour
    # any pixel that is "far" enough away is assumed to be ink
    for c in range(lower_X+1,upper_X):
        for r in range(lower_Y+1,upper_Y):
            pixel_colour = tuple(image[r,c])
            # pixel_colour = int(image[r,c])
            # print pixel_colour
            if pixel_colour not in colours:
                colours[pixel_colour] = 1
            else:
                colours[pixel_colour] += 1

    most_common_colour,_ = sorted(colours.items(),key = lambda x:x[1],reverse=True)[0]
    pts = []

    # extract the ink pixels
    for c in range(lower_X+1,upper_X):
        for r in range(lower_Y+1,upper_Y):
            pixel_colour = tuple(image[r,c])

            dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(pixel_colour,most_common_colour)]))

            # print dist

            if dist > 40:
                # todo - get this flipping figured out
                pts.append((c,r))

    # return if we have an empty cell
    if pts == []:
        return

    # convert to an numpy array because ... we have to
    pts = np.asarray(pts)

    # do dbscan
    db = DBSCAN(eps=3, min_samples=20).fit(pts)
    labels = db.labels_
    unique_labels = set(labels)

    # each cluster should hopefully correspond to a different digit

    digit_probabilities = []

    for k in unique_labels:
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

        print (min_y,max_y+1)
        print (min_x,max_x+1)

        # todo - this will probably include noise-pixels, so we need to redo this
        template = image[np.ix_(r, c)]

        digit_image = Image.fromarray(np.uint8(np.asarray(template)))
        # plt.show()
        # cv2.imwrite("/home/ggdhines/aa.png",np.uint8(np.asarray(template)))
        # raw_input("template extracted")
        # continue

        digit_image = digit_image.resize((width,height),Image.ANTIALIAS)
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

        for y in range(len(digit_array)):
            for x in range(len(digit_array[0])):
                # dist1 = math.sqrt(sum([(a-b)**2 for (a,b) in zip(digit_array[y][x],ref1)]))
                # if dist1 > 10:
                # if digit_array[y][x] > 0.4:
                #     plt.plot(x+x_offset,y+y_offset,"o",color="blue")
                # digit_array[y][x] = digit_array[y][x]/255.
                # print digit_array[y][x] - most_common_colour
                dist = math.sqrt(sum([(int(a)-int(b))**2 for (a,b) in zip(digit_array[y][x],most_common_colour)]))


                if dist > 40:#digit_array[y][x] > 10:
                    centered_array[(y+y_offset)*28+(x+x_offset)] = grey_image[y][x]#/float(darkest_pixel)
                    # print "*",
                else:
                    centered_array[(y+y_offset)*28+(x+x_offset)] = 0
                    # print " ",
            # print

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
        centered_array = T.transform(centered_array)
        # print centered_array
        # print clf.predict_proba(centered_array)
        # raw_input("enter something")
        t = clf.predict_proba(centered_array)
        # print t
        # print list(t)
        # print t[0]
        # print

        digit_probabilities.append(max(t[0]))

    print digit_probabilities
    raw_input("enter something")


if __name__ == "__main__":
    for f_count,f_name in enumerate(log_pages):
        if not f_name.endswith(".png"):
            continue
        # f_name = "Bear-AG-29-1941-0493_0_4.png"
        print f_name

        im = Image.open(image_directory+f_name)
        # im = im.convert('L')#.convert('LA')
        # image = np.asarray(im)

        __extract__(im)

        # break