#!/usr/bin/env python
__author__ = 'greg'
from copy import deepcopy
import numpy as np

import datetime

#each image that was sent out contained several lines to be transcribed
#give the index of the image and the line
imageIndex = 1
lineIndex = 1

#which input files do you want to read in- need to be of the form transcribe$i.txt
#right now can handle at most only 3 or so files - NP-hard problem. Future work might be to make this
#code more scalable - probably with branch and bound approach
inputFiles = [2,3,4]

lines = {}

#read in the files - right now just set up to work on Greg's computer
for i in inputFiles:
    currentImage = 0
    currentLine = -1
    with open("/home/greg/Databases/transcribe/transcribe"+str(i)+".txt","rb") as f:
        for l in f.readlines():
            if l == "\n":
                currentImage += 1
                currentLine = -1
                continue

            currentLine += 1
            if not(currentImage,currentLine) in lines:
                lines[(currentImage,currentLine)] = [l[:-1]]
            else:
                lines[(currentImage,currentLine)].append(l[:-1])

#find the length of each string
stringLength = [len(l)+1 for l in lines[(imageIndex,lineIndex)]]

#record the length of longest common subsequence - assumes unique LCS
#that is there might be more than one longest common subsequence. Have encountered this in practice
#but the strings are usually similar enough that it doesn't matter which LCS you choose
dynamicMatrix = np.zeros(stringLength)

#keep track of the time needed to do the calculation - mainly just because it is NP-hard
#want to know how big the input is you can handle
time1 = datetime.datetime.now()

#the following is the dynamic programming approach as shown on the Wikipedia page for longest common subsequence
traceMatrices = [np.zeros(stringLength) for l in lines[(imageIndex,lineIndex)]]
transcribed = lines[(imageIndex,lineIndex)]
#need to wrap so that numpy is happy
#index for iterating over all tuples of characters - one from each string
currentIndex = [[1,] for w in transcribed]

#dynamic programming approach - basically just filling in a matrix as we go
while True:
    characters = [transcribed[j][currentIndex[j][0]-1] for j in range(len(transcribed))]

    #if we have a match across all strings
    if min(characters) == max(characters):
        #diagional move
        newIndex = [[i[0]-1,] for i in currentIndex]

        #the longest common previous subsequence is a diagonal move backwards
        for j in range(len(newIndex)):
            traceMatrices[j][currentIndex] = newIndex[j]
        dynamicMatrix[currentIndex] = dynamicMatrix[newIndex] + 1

    else:
        #either a up or sideways move
        #find which is the maximum - assume unique
        maxLength = -1
        axis = None

        #find out where the previous LCS is - either up or sideways
        for j in range(len(currentIndex)):
            #move backwards along this axis
            newIndex = deepcopy(currentIndex)
            newIndex[j][0] += -1

            if dynamicMatrix[newIndex][0] > maxLength:
                maxLength = dynamicMatrix[newIndex][0]
                axis = j

        newIndex = deepcopy(currentIndex)
        newIndex[axis][0] += -1

        for j in range(len(newIndex)):
            traceMatrices[j][currentIndex] = newIndex[j]
        dynamicMatrix[currentIndex] = dynamicMatrix[newIndex]


    #iterate to the next tuple of characters
    for j in range(0,len(currentIndex)):
        currentIndex[j][0] += 1
        if currentIndex[j][0] == (len(transcribed[j])+1):
            currentIndex[j][0] = 1
        else:
            break



    if currentIndex == [[1,] for l in transcribed]:
        break

#check to see if the last tuple of characters is a match
lastCharacter = [t[-1] for t in transcribed]
s = [[] for t in transcribed]
if min(lastCharacter) == max(lastCharacter):
    for i,w in enumerate(transcribed):
        s[i].append(len(w)-1)

#read out the LCS by travelling backwards (up, left or diagonal) through the matrix
endPoint = [[-1,] for j in transcribed]
cell = [[int(traceMatrices[j][endPoint]),] for j in range(len(transcribed))]


while cell != [[0,] for j in range(len(transcribed))]:
    newcell = [[int(traceMatrices[j][cell][0]),] for j in range(len(transcribed))]

    #if we have a diagonal move - this corresponds to a point in the LCS
    allChange = not(False in [a!=b for (a,b) in zip(cell,newcell)])
    if allChange:
        for j in range(len(transcribed)):
            s[j].append(newcell[j][0])

    cell = newcell

#print out how long this took
time2 = datetime.datetime.now()
print time2-time1

#print out the LCS in green, all other characters in red
from termcolor import colored
for j,tran in enumerate(transcribed):
    for i,c in enumerate(tran):
        if i in s[j]:
            print colored(c,'green'),
        else:
            print colored(c,'red'),

    print