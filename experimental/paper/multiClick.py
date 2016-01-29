#!/usr/bin/env python
__author__ = 'greg'
import matplotlib.pyplot as plt

errors = []
notErrors = []



with open("/Users/greg/Databases/errors","rb") as f:
    for l in f.readlines():
        pts = [float(p) for p in l[:-1].split(" ")]
        x = max(pts[0],pts[1])
        y = min(pts[0],pts[1])
        errors.append((x,y))
        #plt.plot(x,y,"o",color="blue")

with open("/Users/greg/Databases/doubleClick","rb") as f:
    for l in f.readlines():
        pts = [float(p) for p in l[:-1].split(" ")]
        x = max(pts[0],pts[1])
        y = min(pts[0],pts[1])
        if not (x,y) in errors:
            notErrors.append((x,y))
        #plt.plot(x,y,"o",color="red")

#plt.show()
TE = len([1 for (x,y) in errors if y >= 2])
ME = len([1 for (x,y) in errors if y < 2])

FE = len([1 for (x,y) in notErrors if y >= 2])
noE = len([1 for (x,y) in notErrors if y < 2])

print TE,ME
print FE,noE

print TE/float(TE+FE),FE/float(TE+FE)
print ME/float(ME+noE),noE/float(ME+noE)