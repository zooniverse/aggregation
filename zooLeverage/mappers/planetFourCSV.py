#!/usr/bin/env python
__author__ = 'ggdhines'
import sys

for l,line in enumerate(sys.stdin):
    if l == 0:
        continue

    # remove leading and trailing whitespace
    words = line.split(",")

    subject_zoonvirse_id = words[2]
    user_name = words[4]
    shapeType = words[5]

    x = words[8]
    y = words[9]

    if (shapeType == "blotch") or (shapeType == "fan"):
        if shapeType == "blotch":
            w = words[12]
            h = words[13]
            layer = "0"
        else:
            w = words[14]
            h = words[16][:-1]
            layer = "1"
        r = words[15]

        print subject_zoonvirse_id + "\t" + user_name + ",ellipse," + layer + "," + x + "," + y + "," + w + "," + h + "," + r

    elif shapeType == "interesting":
        print subject_zoonvirse_id + "\t" + user_name + ",point,2," + x + "," + y
    else:
        assert(False)
