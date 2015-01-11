#!/usr/bin/env python
__author__ = 'greghines'

count = 0

with open("/home/greg/Documents/condor_gold","rb") as f:
    for l in f.readlines():
        if not(" " in l):
            count += 1

print count