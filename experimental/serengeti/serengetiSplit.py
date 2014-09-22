#!/usr/bin/env python

import sys

start = int(sys.argv[1])
stop = int(sys.argv[2])

f = open("/Users/greghines/Downloads/2014-05-18_serengeti_classifications.csv","rb")

for i,l in enumerate(f):
    if i == 0:
        print l[:-1]

    if i == (stop+1):
        break

    if i >= start:
        print l[:-1]