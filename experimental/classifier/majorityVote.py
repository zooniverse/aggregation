#!/usr/bin/env python
from __future__ import print_function
__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np
import csv
import math
import pymongo
import os
from copy import deepcopy


class MajorityVote:
    def __init__(self,subjectNodes,userNodes):
        self.subjectNodes = subjectNodes
        self.userNodes = userNodes
        self.alpha = 0.6

    def __classify__(self):
        correct = 0
        total = 0.

        for subject in self.subjectNodes:
            votes = subject.__getVotes__()
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





