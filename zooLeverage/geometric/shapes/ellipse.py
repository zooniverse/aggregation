__author__ = 'ggdhines'
import random
import math


numSamples = 100


def DBSCANmap(ellipse):
    return ellipse[0],ellipse[1]

def withinEllipse(ellipse,pt):
    x,y,w,h,r = ellipse
    r = math.radians(r)

    X,Y = pt

    #center the ellipse at 0,0
    X = X - x
    Y = Y - y

    d =math.pow(X*math.cos(r) + Y*math.sin(r),2)/math.pow(w,2) + math.pow(X*math.sin(r) + Y*math.cost(r),2)/math.pow(h,2)

    return d <= 1

def similarity(ellipse1, ellipse2):
    ####assuming width and height are half !!!!!
    x1,y1,w1,h1,r1 = ellipse1
    x2,y2,w2,h2,r2 = ellipse2

    #find a box containing all both of these ellipses - check to see if they don't overlap
    #if they don't, just return 0

    minX1 = x1 - max(w1,h1)
    minX2 = x2 - max(w2,h2)
    minX = min(minX1,minX2)

    maxX1 = x1 + max(w1,h1)
    maxX2 = x2 + max(w2,h2)
    maxX = max(maxX1,maxX2)

    #won't catch all non-overlapping ellipses but hopefully will get most
    if (minX1 >= maxX2) or (minX2 >= maxX1):
        return 0

    #now repeat for Y coordinates
    minY1 = y1 - max(w1,h1)
    minY2 = y2 - max(w2,h2)
    minY = min(minY1,minY2)

    maxY1 = y1 + max(w1,h1)
    maxY2 = y2 + max(w2,h2)
    maxY = max(maxY1,maxY2)

    #won't catch all non-overlapping ellipses but hopefully will get most
    if (minY1 >= maxY2) or (minY2 >= maxY1):
        return 0

    #hopefully if we have got this far, the ellipses are overlapping
    #start sampling from the box
    intersectionPts = 0
    for sampleIndex in range(numSamples):
        x = random.uniform(minX,maxX)
        y = random.uniform(minY,maxY)

        #does the point fall within the first ellipse?
        if withinEllipse(ellipse1,(x,y)) and withinEllipse(ellipse2,(x,y)):
            intersectionPts += 1

    #find the area of the intersection
    intersectionArea = intersectionPts/float(numSamples)*(maxX - minX)*(maxY - minY)

    ellipseArea1 = math.pi*w1*h1
    ellipseArea2 = math.pi*w2*h2

    return min(intersectionArea/ellipseArea1,intersectionArea/ellipseArea2)



