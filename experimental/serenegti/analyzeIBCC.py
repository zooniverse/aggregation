#!/usr/bin/env python
import csv
__author__ = 'ggdhines'

f = open("/home/ggdhines/Databases/serengeti/output.csv", "rb")
reader = csv.reader(f, delimiter=" ")

f2 = open("/home/ggdhines/Databases/serengeti/expertIBCC.csv", "rb")
reader2 = csv.reader(f2, delimiter=",")

numPictures = 0.
alpha = 0.5
numMatches = 0

for userRow, expertRow in zip(reader, reader2):
    pictureID_u = int(float(userRow[0]))
    userProb = float(userRow[2])

    if userProb >= alpha:
        uClassification = 1
    else:
        uClassification = 0

    pictureID_e = int(float(expertRow[0]))
    eClassification = float(expertRow[1])
    assert(pictureID_e == pictureID_u)

    numPictures += 1
    if uClassification == eClassification:
        numMatches += 1

print numMatches/numPictures