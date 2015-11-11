import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from sklearn.cluster import DBSCAN

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

for f_count,f_name in enumerate(log_pages):
    if not f_name.endswith(".png"):
        continue

    im = Image.open(image_directory+f_name)
    im = im.convert('L')#.convert('LA')
    image = np.asarray(im)

    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=10,
                                     line_gap=0)

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

    print unique_labels
    print labels

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

            if percent > 0.5:
                for ((x1,y1),(x2,y2)) in segments:
                    ax3.plot((x1,x2), (y1,y2),color=col)

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

    ax3.set_title('Probabilistic Hough')
    ax3.set_axis_off()
    ax3.set_adjustable('box-forced')
    plt.show()