#!/usr/bin/env python
__author__ = 'greg'
from copy import deepcopy
import numpy as np

import datetime

imageIndex = 1
lineIndex = 1

inputFiles = [2,3,4]

lines = {}

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

#print lines[(0,0)]



stringLength = [len(l)+1 for l in lines[(imageIndex,lineIndex)]]

#record the length of longest common subsquence - assumes unique
dynamicMatrix = np.zeros(stringLength)

#print dynamicMatrix
time1 = datetime.datetime.now()

traceMatrices = [np.zeros(stringLength) for l in lines[(imageIndex,lineIndex)]]
transcribed = lines[(imageIndex,lineIndex)]
#need to wrap so that numpy is happy
currentIndex = [[1,] for w in transcribed]
while True:
    characters = [transcribed[j][currentIndex[j][0]-1] for j in range(len(transcribed))]


    if min(characters) == max(characters):
        #we have a match across all strings
        #diagional move
        newIndex = [[i[0]-1,] for i in currentIndex]

        for j in range(len(newIndex)):
            traceMatrices[j][currentIndex] = newIndex[j]
        dynamicMatrix[currentIndex] = dynamicMatrix[newIndex] + 1

    else:
        #either a up or sideways move
        #find which is the maximum - assume unique
        maxLength = -1
        axis = None

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

    for j in range(0,len(currentIndex)):
        currentIndex[j][0] += 1
        if currentIndex[j][0] == (len(transcribed[j])+1):
            currentIndex[j][0] = 1
        else:
            break



    if currentIndex == [[1,] for l in transcribed]:
        break

lastCharacter = [t[-1] for t in transcribed]
s = [[] for t in transcribed]
if min(lastCharacter) == max(lastCharacter):
    for i,w in enumerate(transcribed):
        s[i].append(len(w)-1)

endPoint = [[-1,] for j in transcribed]
#cell = [[int(traceMatrices[0][-1][-1]),],[int(traceMatrices[1][-1][-1]),]]
cell = [[int(traceMatrices[j][endPoint]),] for j in range(len(transcribed))]
#print len(lines[(0,0)][0])
#print cell


while cell != [[0,] for j in range(len(transcribed))]:
    #X = int(traceMatrices[0][cell][0])
    #Y = int(traceMatrices[1][cell][0])
    newcell = [[int(traceMatrices[j][cell][0]),] for j in range(len(transcribed))]
    #print newcell

    allChange = not(False in [a!=b for (a,b) in zip(cell,newcell)])
    if allChange:
        for j in range(len(transcribed)):
            s[j].append(newcell[j][0])

    cell = newcell

time2 = datetime.datetime.now()
print time2-time1

from termcolor import colored
for j,tran in enumerate(transcribed):
    for i,c in enumerate(tran):
        if i in s[j]:
            print colored(c,'green'),
        else:
            print colored(c,'red'),

    print