#!/usr/bin/env python

f1 = open("/Users/greg/Databases/base_ibcc.out","rb")
f2 = open("/Users/greg/Databases/merged_ibcc.out","rb")

while True:
    l1 = f1.readline()


    if l1 == "":
        break

    lineIndex = int(float(l1.split(" ")[0]))
    if lineIndex != 37:
        l2 = f2.readline()

        p1 = float(l1.split(" ")[2])
        p2 = float(l2.split(" ")[2])
        print p1-p2
    else:
        break
