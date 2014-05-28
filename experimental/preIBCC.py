#!/usr/bin/env python

__author__ = 'ggdhines'
import csv
import bisect

animal = "wildebeest"
f = open("/home/ggdhines/aws/2014-05-25_serengeti_classifications.csv",'rb')
reader = csv.reader(f,delimiter = ",")

userList = []
photoList = []

userDict = {}

next(reader,None)
i = 0
for row in reader:
    if row[3] == "tutorial":
        continue

    i += 1


    userStr = row[1]
    photoStr = row[2]
    speciesStr = row[11]

    try:
        userID = userList.index(userStr)
    except ValueError:
        userList.append(userStr)
        userID = len(userList) -1

    try:
        user = userDict[userID]
    except KeyError:
        userDict[userID] = {}
        user = userDict[userID]


    try:
        photoID = photoList.index(photoStr)
    except ValueError:
        photoList.append(photoStr)
        photoID = len(photoList) - 1

    #has the user already tagged this photo?
    if photoID in user:
        user[photoID] = (speciesStr == animal) or user[photoID]
    else:
        user[photoID] = (speciesStr == animal)


for userID in userDict:
    user = userDict[userID]
    for photoID in user:
        if user[photoID] == True:
            f = 1
        else:
            f = 0
        print str(userID) + "," + str(photoID) + "," + str(f)

