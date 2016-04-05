"""
helper functions for extracting necessary parameters for different types of markings
also for dimensionality reduction
"""
from __future__ import print_function
import math
import random
import unicodedata
import re
import sys

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
        warning(marking)
        raise

    if (x<0)or(y<0)or(x > image_dimensions[0]) or(y>image_dimensions[1]):
        # warning( "marking probably outside of image")
        # warning( marking)
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
        if(x2 > image_dimensions[1]) or(y2>image_dimensions[0]):
            raise InvalidMarking(marking)

    return (x,y),(x,y2),(x2,y2),(x2,y)


def relevant_circle_params(marking,image_dimensions):
    return marking["x"],marking["y"],marking["r"]


def relevant_ellipse_params(marking,image_dimensions):
    return marking["x"],marking["y"],marking["rx"],marking["ry"],marking["angle"]


def relevant_polygon_params(marking,image_dimensions):
    points = marking["points"]
    return tuple([(p["x"],p["y"]) for p in points])


def relevant_bezier_params(marking,image_dimensions):
    points = marking["points"]
    output = [(points[0]["x"], points[0]["y"])]
    def get_points_along_curve(pStart, pControl, pEnd, output):
        # t is position along curve, add more to get better approximation
        # this splits each segment into 5 straight lines
        for t in [0.2, 0.4, 0.6, 0.8, 1]:
            x = (1 - t) * (1 - t) * pStart["x"] + 2 * (1 - t) * t * pControl["x"] + t * t * pEnd["x"]
            y = (1 - t) * (1 - t) * pStart["y"] + 2 * (1 - t) * t * pControl["y"] + t * t * pEnd["y"]
            output.append((x, y))
    N = len(points)
    for idx in range(0, N, 2):
        # check if closed automatically (odd length to points) and add in the final point
        if idx+1 == N:
            output.append((points[idx + 1]["x"], points[idx + 1]["y"]))
        # check if this is the last segment and close with the first point
        elif idx+2 == N:
            get_points_along_curve(points[idx], points[idx + 1], points[0], output)
            # don't duplicate the inital point
            output = output[:-1]
        else:
            get_points_along_curve(points[idx], points[idx + 1], points[idx + 2], output)
    return output


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

def relevant_text_params(marking,image_dimensions):
    """
    extract the relevant params from the the transcription marking
    note that the text is the last item - which means we can treat the results
    pretty much like a line segment - which it mostly is
    :param marking:
    :param image_dimensions:
    :return:
    """
    if ("startPoint" not in marking) or ("endPoint" not in marking):
        raise InvalidMarking(marking)
    x1 = marking["startPoint"]["x"]
    y1 = marking["startPoint"]["y"]
    x2 = marking["endPoint"]["x"]
    y2 = marking["endPoint"]["y"]

    if min(x1,x2,y1,y2) < 0:
        raise InvalidMarking(marking)

    if "text" not in marking:
        raise InvalidMarking(marking)

    text = marking["text"]

    if "variants" in marking:
        variants = [v for v in marking["variants"] if v != ""]
    else:
        variants = []

    if x1 <= x2:
        return x1,x2,y1,y2,text,variants
    else:
        return x2,x2,y2,y1,text,variants


def text_line_reduction(line_segments):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """
    reduced_markings = []

    for line_seg in line_segments:
        # last value is variants which is only relevant to Folger
        x1,y1,x2,y2,text,_ = line_seg

        x2 += random.uniform(-0.0001,0.0001)
        x1 += random.uniform(-0.0001,0.0001)

        dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

        try:
            tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
            theta = math.atan(tan_theta)
        except ZeroDivisionError:
            theta = math.pi/2.

        reduced_markings.append((dist,theta,text))

    return reduced_markings

def csv_string(string):
    """
    remove or replace all characters which might cause problems in a csv template
    :param str:
    :return:
    """
    if type(string) == unicode:
        string = unicodedata.normalize('NFKD', string).encode('ascii','ignore')
    string = re.sub(' ', '_', string)
    string = re.sub(r'\W+', '', string)

    return string

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)