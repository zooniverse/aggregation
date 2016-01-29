#!/usr/bin/env python
__author__ = 'greghines'

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    values = line.split(",")
    if values[0] == "\"id\"":
        continue

    print str(values[2][1:-1])+"\t" + str(values[11][1:-1])