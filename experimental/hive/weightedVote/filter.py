#!/usr/bin/env python
f = open("/Users/greghines/Databases/serengeti/output","rb")

for lines in f.readlines():
    words = lines.split(",")
    l = [i for i,s in enumerate(words) if s == "\"2012-06-10T22:11:52Z;2012-06-10T22:11:52Z;2012-06-10T22:11:52Z\""]

    print l
