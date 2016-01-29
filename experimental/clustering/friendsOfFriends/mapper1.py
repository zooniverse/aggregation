#!/usr/bin/env python
import sys
import math


# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    if i == 0:
        continue

    words = line.split(",")
    try:
        image = words[2]

    except IndexError:
        sys.stderr.write(line[:-1])
        continue

    if image[-9:-1] == "tutorial":
        continue

    try:
        p0_x = float(words[6][1:-1])
        p0_y = float(words[7][1:-1])

        try:
            p1_x = float(words[8][1:-1])
            p1_y = float(words[9][1:-1])

            try:
                p2_x = float(words[10][1:-1])
                p2_y = float(words[11][1:-1])

                avg_x = (p0_x + p1_x + p2_x)/3.
                avg_y = (p0_y + p1_y + p2_y)/3.
            except ValueError:
                avg_x = (p0_x + p1_x)/2.
                avg_y = (p0_y + p1_y)/2.

        except ValueError:
            avg_x = p0_x
            avg_y = p0_y

    except ValueError:
        avg_x = None
        avg_y = None

    if avg_x != None:
        print image[1:-1] + "\t" + str(avg_x)+"," + str(avg_y)

    #if i > 5000:
    #    break