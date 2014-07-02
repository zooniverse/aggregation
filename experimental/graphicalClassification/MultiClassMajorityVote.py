#!/usr/bin/env python
from __future__ import print_function
__author__ = 'greghines'

class MultiClassMajorityVote:
    def __init__(self,subjectNodes,userNodes):
        self.subjectNodes = subjectNodes
        self.userNodes = userNodes
        self.alpha = 0.6

    def __classify__(self,attributeList):
        for att in attributeList:
            for user in self.userNodes:
                user.__changeClassificationAttributes__(att)

            for subject in self.subjectNodes:
                subject.__changeClassificationAttributes__(att)

                #what alpha value would this subject need to get correct positive?

