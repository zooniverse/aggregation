"""
helper functions for extracting necessary parameters for different types of markings
also for dimensionality reduction
"""
import math
import random


class InvalidMarking(Exception):
    def __init__(self,pt):
        self.pt = pt
    def __str__(self):
        return "invalid marking: " + str(self.pt)

class EmptyPolygon(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return "empty polygon"

# extract the relevant params for different shapes from the json blob
# todo - do a better job of checking to make sure that the marking lies within the image dimension
# todo - also generalize to ROI
def relevant_line_params(marking,image_dimensions):
    # want to extract the params x1,x2,y1,y2 but
    # ALSO make sure that x1 <= x2 and flip if necessary
    x1 = marking["x1"]
    x2 = marking["x2"]
    y1 = marking["y1"]
    y2 = marking["y2"]

    if min(x1,x2,y1,y2) < 0:
        raise InvalidMarking(marking)

    # only do this part if we have been provided dimensions
    if image_dimensions is not None:
        if (max(x1,x2) >= image_dimensions[0]) or (max(y1,y2) >= image_dimensions[1]):
            raise InvalidMarking(marking)

    if x1 <= x2:
        return x1,y1,x2,y2
    else:
        return x2,y2,x1,y1


# the following convert json blobs into sets of values we can actually cluster on
# todo - do a better job with checking whether the markings fall within the image_dimensions
def relevant_point_params(marking,image_dimensions):
    # todo - this has to be changed
    image_dimensions = 100000,100000
    if (marking["x"] == "") or (marking["y"] == ""):
        raise InvalidMarking(marking)

    try:
        x = float(marking["x"])
        y = float(marking["y"])
    except ValueError:
        print marking
        raise

    if (x<0)or(y<0)or(x > image_dimensions[0]) or(y>image_dimensions[1]):
        print "marking probably outside of image"
        print marking
        raise InvalidMarking(marking)

    return x,y


def relevant_rectangle_params(marking,image_dimensions):
    x = float(marking["x"])
    y = float(marking["y"])
    x2 = x + float(marking["width"])
    y2 = y + float(marking["height"])

    # not sure how nan can happen but apparently it can
    if math.isnan(x) or math.isnan(y) or math.isnan(x2) or math.isnan(y2):
        raise InvalidMarking(marking)
    if min(x,y,x2,y2) < 0:
        raise InvalidMarking(marking)


    if (float(marking["width"]) == 0) or (float(marking["height"]) == 0):
        raise EmptyPolygon()

    if (image_dimensions is not None) and (image_dimensions != (None,None)):
        if(x2 > image_dimensions[0]) or(y2>image_dimensions[1]):
            raise InvalidMarking(marking)

    return (x,y),(x,y2),(x2,y2),(x2,y)


def relevant_circle_params(marking,image_dimensions):
    return marking["x"],marking["y"],marking["r"]


def relevant_ellipse_params(marking,image_dimensions):
    return marking["x"],marking["y"],marking["rx"],marking["ry"],marking["angle"]


def relevant_polygon_params(marking,image_dimensions):
    points = marking["points"]
    return tuple([(p["x"],p["y"]) for p in points])


def hesse_line_reduction(line_segments):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """

    reduced_markings = []

    for line_seg in line_segments:
        x1,y1,x2,y2 = line_seg[:4]

        x2 += random.uniform(-0.0001,0.0001)
        x1 += random.uniform(-0.0001,0.0001)

        dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

        try:
            tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
            theta = math.atan(tan_theta)
        except ZeroDivisionError:
            theta = math.pi/2.

        reduced_markings.append([dist,theta])

        # for cases where we have lines of text, don't want to ignore those
        if len(line_seg) > 4:
            reduced_markings[-1].append(line_seg[4])

    return reduced_markings
