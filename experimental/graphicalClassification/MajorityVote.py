#!/usr/bin/env python
from __future__ import print_function
__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np


class MajorityVote:
    def __init__(self,subjectNodes,userNodes):
        self.subjectNodes = subjectNodes
        self.userNodes = userNodes
        self.alpha = 0.6

    def __roc__(self):
        pos = []
        neg = []
        for subject in self.subjectNodes:
            votes = subject.__getVotes__()
            goldStandard = subject.__getGoldStandard__()

            if goldStandard == 0:
                pos.append(votes[0])
                neg.append(votes[1])
            else:
                pos.append(votes[1])
                neg.append(votes[0])

        pos.sort()
        neg.sort()

        pEnumerated = list(enumerate(pos))
        pEnumerated.reverse()

        nEnumerated = list(enumerate(neg))
        nEnumerated.reverse()

        lx = []
        ly = []

        for alpha in np.arange(0,1.01,0.01):
            found = False
            for pIndex,pAlpha in pEnumerated:
                if pAlpha <= alpha:
                    found = True
                    break
            assert(found)

            pPercent = pIndex/float(len(pos))

            found = False
            for nIndex,nAlpha in nEnumerated:
                if nAlpha <= alpha:
                    found = True
                    break
            assert(found)

            nPercent = nIndex/float(len(neg))

            lx.append(pPercent)
            ly.append(nPercent)

        return lx,ly





    def __classify__(self,classList):
        correct = 0
        total = 0.

        for subject in self.subjectNodes:
            votes = subject.__getVotes__(classList)
            if votes[1] >= self.alpha:
                aggregateClassification = 1
            else:
                aggregateClassification = 0

            assert(len(votes) == 2)
            goldStandard = subject.__getGoldStandard__()

            if goldStandard == aggregateClassification:
                correct += 1
            total += 1

        print(correct/total)





