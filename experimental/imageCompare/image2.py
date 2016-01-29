#!/usr/bin/env python
import scipy as sp
from scipy.misc import imread
from scipy.signal.signaltools import correlate2d as c2d


def get(fName):
    # get JPG image as Scipy array, RGB (3 layer)
    data = imread(fName)
    # convert to grey-scale using W3C luminance calc
    data = sp.inner(data, [299, 587, 114]) / 1000.0
    # normalize per http://en.wikipedia.org/wiki/Cross-correlation
    return (data - data.mean()) / data.std()

im1 = get("/home/ggdhines/Databases/serengeti/temp2/im2.jpg")

im2 = get("/home/ggdhines/Databases/serengeti/temp2/im3.jpg")
im3 = get("/home/ggdhines/Databases/serengeti/temp2/im1.jpg")
print "analyzing"
c11 = c2d(im1, im1, mode='same')
print c11.max()
c12 = c2d(im1, im2, mode='same')
print c12.max()
c13 = c2d(im1, im3, mode='same')


print c13.max()