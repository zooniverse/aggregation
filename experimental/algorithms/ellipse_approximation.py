__author__ = 'greg'
from matplotlib.patches import Ellipse
from pylab import figure, show, rand
import math
import numpy
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

fig = figure()
ax = fig.add_subplot(111, aspect='equal')

x_0,y_0,rx,ry,angle = 266.916687012, 114, 62.2253967444, 22.6715680975,45

e = Ellipse((x_0,y_0), rx*2, ry*2, angle)
ax.add_artist(e)
ax.set_xlim(150, 350)
ax.set_ylim(0, 250)

alpha = angle * math.pi/180.

X = []
Y = []

for theta in numpy.linspace(0,2*math.pi,6):
    x = x_0 + rx*math.cos(theta)*math.cos(alpha) - ry*math.sin(theta)*math.sin(alpha)
    y = y_0 + rx*math.cos(theta)*math.sin(alpha) + ry*math.sin(theta)*math.cos(alpha)
    # print x,y
    plt.plot(x,y,"o")

    X.append(x)
    Y.append(y)

p = Polygon(zip(X,Y))
print p.area/(math.pi*rx*ry)
# print math.pi*rx*ry
show()