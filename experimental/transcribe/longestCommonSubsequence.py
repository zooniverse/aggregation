#!/usr/bin/env python
__author__ = 'greg'
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import itertools

#important cases
# ('gold but black', 'gold out flack', 'Gold ')
# [4, 3, 2, 1]
# [8, 3, 2, 1]
# [4, 3, 2, 1]
# [False, True, False]
# [['g', 'old ', 'but black'], ['g', 'old', ' out', ' ', 'flack'], ['G', 'old ']]

# for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
    code_directory = base_directory + "/github"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"

else:
    base_directory = "/home/greg"
    code_directory = base_directory + "/github"

#each image that was sent out contained several lines to be transcribed
#give the index of the image and the line
imageIndex = 1
lineIndex = 4

#which input files do you want to read in- need to be of the form transcribe$i.txt
#right now can handle at most only 3 or so files - NP-hard problem. Future work might be to make this
#code more scalable - probably with branch and bound approach
inputFiles = [1,2,3,4,5]

def load_file(fname):
    currentImage = 0
    currentLine = -1

    individual_transcriptions = {}

    with open(fname,"rb") as f:
        for l in f.readlines():
            if l == "\n":
                currentImage += 1
                currentLine = -1
                continue

            currentLine += 1
            individual_transcriptions[(currentImage,currentLine)] = l[:-1]

    return individual_transcriptions


def lcs(lines):
    #find the length of each string
    stringLength = [len(l)+1 for l in lines]

    #record the length of longest common subsequence - assumes unique LCS
    #that is there might be more than one longest common subsequence. Have encountered this in practice
    #but the strings are usually similar enough that it doesn't matter which LCS you choose
    dynamicMatrix = np.zeros(stringLength)

    #keep track of the time needed to do the calculation - mainly just because it is NP-hard
    #want to know how big the input is you can handle
    time1 = datetime.datetime.now()

    #the following is the dynamic programming approach as shown on the Wikipedia page for longest common subsequence
    traceMatrices = [np.zeros(stringLength) for l in lines]
    #transcribed = lines[(imageIndex,lineIndex)]
    #need to wrap so that numpy is happy
    #index for iterating over all tuples of characters - one from each string
    currentIndex = [[1,] for w in lines]

    #dynamic programming approach - basically just filling in a matrix as we go
    while True:
        characters = [lines[j][currentIndex[j][0]-1] for j in range(len(lines))]

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
            if currentIndex[j][0] == (len(lines[j])+1):
                currentIndex[j][0] = 1
            else:
                break



        if currentIndex == [[1,] for l in lines]:
            break

    #check to see if the last tuple of characters is a match
    lastCharacter = [t[-1] for t in lines]
    s = [[] for t in lines]
    lcs_length = 0
    if min(lastCharacter) == max(lastCharacter):
        lcs_length += 1
        for i,w in enumerate(lines):
            s[i].append(len(w)-1)

    #read out the LCS by travelling backwards (up, left or diagonal) through the matrix
    endPoint = [[-1,] for j in lines]
    cell = [[int(traceMatrices[j][endPoint]),] for j in range(len(lines))]




    while cell != [[0,] for j in range(len(lines))]:
        newcell = [[int(traceMatrices[j][cell][0]),] for j in range(len(lines))]

        #if we have a diagonal move - this corresponds to a point in the LCS
        allChange = not(False in [a!=b for (a,b) in zip(cell,newcell)])
        if allChange:
            lcs_length += 1
            for j in range(len(lines)):
                s[j].append(newcell[j][0])

        cell = newcell

    #print out how long this took
    time2 = datetime.datetime.now()

    # use the first string to actually create the LCS
    lcs_string = ""
    for i,c in enumerate(lines[0]):
        if i in s[0]:
            lcs_string += c

    # print time2-time1
    #
    #print out the LCS in green, all other characters in red
    results = [[] for l in lines]
    at_lcs = [None for l in lines]
    agreement = []

    s = [sorted(s_temp) for s_temp in s]

    LCStuples = [[s[j][i] for j in range(len(lines))] for i in range(len(s[0]))]

    LCSsequences = [[LCStuples[0]]]
    for i in range(1,len(s[0])):
        max_character_jump = max([(s[j][i] - s[j][i-1]) for j in range(len(lines))])
        if max_character_jump > 1:
            LCSsequences.append([])
        LCSsequences[-1].append(LCStuples[i])


    segments = {}

    lcs_string = ""

    for j in range(len(lines)):
        currentIndex = 0
        results = []
        for sequenceIndex,nextSequence in enumerate(LCSsequences):
            firstLCSChacter = nextSequence[0][j]
            lastLCSCharacter = nextSequence[-1][j]
            l = lines[j][currentIndex:firstLCSChacter]
            if l != "":
                if not (2*sequenceIndex) in segments:
                    segments[2*sequenceIndex] = [l]
                else:
                    segments[2*sequenceIndex].append(l)

            # now extra the LCS - we only need to do this once, since every one is in agreement
            if j == 0:
                l = lines[0][firstLCSChacter:lastLCSCharacter+1]
                segments[2*sequenceIndex+1] = l[:]

                lcs_string += l

            currentIndex = lastLCSCharacter + 1

        l = lines[j][currentIndex:]
        if l != "":
            if not (2*(sequenceIndex+1)) in segments:
                segments[2*(sequenceIndex+1)] = [l]
            else:
                segments[2*(sequenceIndex+1)].append(l)
            # results.append((l,-sequenceIndex-2))
            # segments.add(-sequenceIndex-2)

    return lcs_string,segments

    from termcolor import colored
    for j,tran in enumerate(lines):
        print s[j]
        for i,c in enumerate(tran):
            if i in s[j]:
                now_at_csl = True
                #print colored(c,'green'),
            else:
                now_at_csl = False
                #print colored(c,'red'),
            if now_at_csl != at_lcs[j]:
                results[j].append("")

                if j ==  0:
                    agreement.append(now_at_csl)
            at_lcs[j] = now_at_csl
            results[j][-1] += c
        #print

    return lcs_string,agreement,results

transcriptions = {}
#read in the files - right now just set up to work on Greg's computer
for i in inputFiles:
    fname = base_directory+"/Databases/transcribe/transcribe"+str(i)+".txt"
    individual_transcriptions = load_file(fname)

    for key,line in individual_transcriptions.items():
        if not(key in transcriptions):
            transcriptions[key] = [line]
        else:
            transcriptions[key].append(line)

gold_fname = base_directory+"/Dropbox/goldTranscriptions.txt"
gold_transcriptions = load_file(gold_fname)

X = []
Y = []

Xstd = []
Ystd = []

# for imageIndex in range(3):
#     for lineIndex in range(5):
#
#         a = []
#         c = []
#
#         for l in transcriptions[(imageIndex,lineIndex)]:
#             lcs_string= lcs([gold_transcriptions[(imageIndex,lineIndex)],l])
#             accuracy = len(lcs_string)/float(len(l))
#             completeness = len(lcs_string)/float(len(gold_transcriptions[(imageIndex,lineIndex)]))
#             a.append(accuracy)
#             c.append(completeness)
#
#         X.append(np.mean(a))
#         Y.append(np.mean(c))
#
#         Xstd.append(np.std(a,ddof=1))
#         Ystd.append(np.std(c,ddof=1))
#
# print X
# print np.mean(X)
# print np.mean(Y)

# plt.errorbar(X,Y,xerr=Xstd,yerr=Ystd,fmt=".")
# plt.xlabel("Accuracy (w/ standard dev.)")
# plt.ylabel("Completeness (w/ standard dev.)")
# plt.xlim((0,1.1))
# plt.ylim((0,1.1))
# plt.show()

def majority_vote(segments,num_users=3):
    new_lcs = ""
    for segmentIndex in sorted(list(segments)):
        # is this segment part of the LCS (if so, don't worry about merging)
        if segmentIndex%2 == 0:
            votes = {}

            for transcription in segments[segmentIndex]:
                if transcription in votes:
                    votes[transcription] += 1
                else:
                    votes[transcription] = 1

            if max(votes.values()) == 2:
                most_likely = max(votes.items(),key = lambda x:x[1])
                new_lcs += most_likely[0]

        else:
            new_lcs += segments[segmentIndex]
        # otherwise - merge
        # grab all of those segments with the the correct segment index

    return new_lcs


def findsubsets(S,m):
    return set(itertools.combinations(S, m))

for imageIndex in range(3):
    for lineIndex in range(5):
        print (imageIndex,lineIndex)

        a = []
        c = []


        for i,s in enumerate(findsubsets(transcriptions[(imageIndex,lineIndex)],3)):
            lcs_string,segments = lcs(s)
            new_lcs_string = majority_vote(segments)
            lcs_string2 = lcs([gold_transcriptions[(imageIndex,lineIndex)],new_lcs_string])
            accuracy = len(lcs_string2)/float(len(lcs_string))
            completeness = len(lcs_string2)/float(len(gold_transcriptions[(imageIndex,lineIndex)]))
            a.append(accuracy)
            c.append(completeness)
        X.append(np.mean(a))
        Y.append(np.mean(c))

        Xstd.append(np.std(a,ddof=1))
        Ystd.append(np.std(c,ddof=1))

print np.mean(X)
print np.mean(Y)


plt.errorbar(X,Y,xerr=Xstd,yerr=Ystd,fmt=".")
plt.xlabel("Accuracy (w/ standard dev.)")
plt.ylabel("Completeness (w/ standard dev.)")
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.show()


#random.sample(transcriptions[(imageIndex,lineIndex)],3)
#lcs_string = lcs(transcriptions[(imageIndex,lineIndex)])
#lcs_string2 = lcs([gold_transcriptions[(imageIndex,lineIndex)],lcs_string])



#print lcs_string2
