#!/usr/bin/env python
__author__ = 'greghines'
import matplotlib.pyplot as plt

xPts = []
yPts = []

with open('/Users/greghines/Databases/gps/smallGPS','r') as f:
    for line in f.readlines():
        x,y = line[:-1].split(',')
        xPts.append(float(x))
        yPts.append(float(y))

fig, ax = plt.subplots()
ax.plot(yPts,xPts,'ro')
plt.show()
