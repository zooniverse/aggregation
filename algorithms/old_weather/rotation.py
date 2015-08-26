__author__ = 'greg'
from math import pi,cos,sin,sqrt

theta = pi/4.

tx = 2
ty = -3

transform = [[cos(theta),-sin(theta),2],[sin(theta),cos(theta),-3],[0,0,1]]
pt = [4,5,1]

pt2 = [sum([a*b for (a,b) in zip(row,pt)]) for row in transform]


def dist(p1, p2):
    return sqrt(sum([(a-b)**2 for (a,b) in zip(p1,p2)]))


def delta(p1,p2):
    return [(a-b) for (a,b) in zip(p1,p2)]

print delta(pt,pt2)

print tx - theta*pt[0]*sin(theta) - theta*pt[1]*cos(theta)
print ty + theta*pt[0]*cos(theta) - theta*pt[1]*sin(theta)