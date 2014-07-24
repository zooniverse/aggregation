#!/usr/bin/env python
__author__ = 'greghines'
import math
from pylab import figure, show
from matplotlib.patches import Ellipse


def minMaxHeight(ellipse):
    x, y, a, b, angle = ellipse
    t = math.atan(-b/a*math.tan(math.radians(angle)))
    print -b/a*math.tan(math.radians(angle))
    print x+ a*math.sin(t)*math.cos(math.radians(angle)) - b*math.cos(t)*math.sin(math.radians(angle))
    t += math.pi/2.
    print x+ a*math.sin(t)*math.cos(math.radians(angle)) - b*math.cos(t)*math.sin(math.radians(angle))

def ellipseIntersection(ellipse1, ellipse2):
    pass

fig = figure()
ax = fig.add_subplot(111, aspect='equal')

minMaxHeight((0.5,0.5,0.6, 0.3, 0.))
ax.add_artist(Ellipse((0.5,0.5),0.3, 0.6, 0))
#ax.add_artist(avgEllipse(clusteredData[0]))
#ax.title('Estimated number of clusters: %d' % n_clusters_)
show()