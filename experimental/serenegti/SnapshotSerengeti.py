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
import sys

if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
else:
    sys.path.append("/Users/greghines/PycharmProjects/reduction/experimental/graphicalClassification")
import MajorityVote
import SubjectNodes
import UserNodes

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class PhotoNode(SubjectNodes.SubjectNode):
    def __init__(self):
        SubjectNodes.SubjectNode.__init__(self)
        self.expertTagged = []

    def __updateExpertClassification__(self,tag):
        self.expertTagged.append(tag)

    def __resetSpeciesList__(self,speciesList):
        self.most_likely_class = None
        self.num_classes = int(math.pow(2,len(speciesList)))
        self.prior_probability = [1/float(self.num_classes) for i in range(self.num_classes)]
        self.updated = True

        if self.expertTagged is not []:
            self.goldStandard = None

            for gIndex, group in enumerate(powerset(speciesList)):
                g_complement = [s for s in speciesList if not(s in group)]

                if not(set(self.expertTagged).intersection(group) == set(group)):
                    pass
                elif not(set(self.expertTagged).intersection(g_complement) == set()):
                    pass
                else:
                    #self.mapped_classification.append(gIndex)
                    self.goldStandard = gIndex
                    break

            assert(self.goldStandard is not None)

class UserNode(UserNodes.UserNode):
    def __init__(self):
        UserNodes.UserNode.__init__(self)
        self.timeStamps = []
        self.animalsTagged = []

        self.speciesList = None

    def __resetSpeciesList__(self,speciesList):
        self.classifications = []

        for pIndex in range(len(self.subjectsViewed)):

            for gIndex, group in enumerate(powerset(speciesList)):
                g_complement = [s for s in speciesList if not(s in group)]

                if not(set(self.animalsTagged[pIndex]).intersection(group) == set(group)):
                    pass
                elif not(set(self.animalsTagged[pIndex]).intersection(g_complement) == set()):
                    pass
                else:
                    #self.mapped_classification.append(gIndex)
                    self.classifications.append(gIndex)
                    break

    def __addAnimal__(self,timeStamp,pNode,animal):
        if pNode in self.subjectsViewed:
            pIndex = self.subjectsViewed.index(pNode)
        else:
            self.subjectsViewed.append(pNode)
            self.animalsTagged.append([])
            self.timeStamps.append(None)
            self.classifications.append(None)
            pIndex = len(self.subjectsViewed)-1

        self.timeStamps[pIndex] = timeStamp

        if animal == "":
            tag = []
        else:
            tag = [animal]

        if pIndex in self.animalsTagged:
            self.animalsTagged[pIndex].extend(tag)
        else:
            self.animalsTagged[pIndex] = tag



    def __getTimeStamp__(self,pNode):
        return self.timeStamps[self.subjectsViewed.index(pNode)]


class SnapshotSerengeti:
    def __init__(self):
        self.userNodes = []
        self.photoNodes = []

        self.user_id_list = []
        self.photo_id_list = []

        self.limit = 5

        if os.path.isdir("/Users/greghines/Databases/serengeti"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            self.baseDir = "/home/ggdhines/Databases/serengeti/"

    def __getPhotoNodes__(self):
        return self.photoNodes

    def __getUserNodes__(self):
        return self.userNodes

    def __readUserClassifications__(self):
        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rU"), delimiter=",")

        i = 0

        next(reader, None)
        for row in reader:
            i+=1
            print(i)
            #if i >= 40000:
            #    break

            userName = row[1]
            photoName = row[2]
            timeStamp = row[4]
            tag = row[11]

            #is this the first this photo has been tagged at all?
            if not(photoName in self.photo_id_list):
                self.photo_id_list.append(photoName)
                pIndex = len(self.photo_id_list)-1
                #self.subjectNodes.append(graphicalEM.SubjectNode(2**len(self.speciesList),photoName))
                self.photoNodes.append(PhotoNode())
                pNode = self.photoNodes[-1]
            else:
                pIndex = self.photo_id_list.index(photoName)
                pNode = self.photoNodes[pIndex]

            #is this the first time we encountered this user at all?
            if not(userName in self.user_id_list):
                self.user_id_list.append(userName)
                uIndex = len(self.user_id_list)-1
                self.userNodes.append(UserNode())
                uNode = self.userNodes[-1]
            else:
                uIndex = self.user_id_list.index(userName)
                uNode = self.userNodes[uIndex]

            #is this the first time this user has tagged this photo?
            if not(pNode.__classifiedBy__(uNode)):
                #have we reached the limit for this particular photo?
                if pNode.__getNumClassifications__() == self.limit:
                    #if so, skip this entry
                    continue
                else:
                    #otherwise add this user's classification
                    #photoDict[photoName][userName] = [timeStamp, newTags]
                    pNode.__addUser__(uNode)
                    uNode.__addAnimal__(timeStamp,pNode,tag)
            else:
                #have we reached the limit for this particular photo?
                if pNode.__getNumClassifications__() == self.limit:
                    currentTimeStamp = uNode.__getTimeStamp__(pNode)
                    #if a user tags multiple animals at once, they will be recorded as separate entries
                    #but HOPEFULLY with the same time stamp, so if the time stamps are the same
                    #add this entry
                    if timeStamp == currentTimeStamp:
                        uNode.__addAnimal__(timeStamp,pNode,tag)
                    else:
                        continue
                else:
                    uNode.__addAnimal__(timeStamp,pNode,tag)





    def __setSpeciesList__(self,speciesList):
        #update all the subject nodes
        for pNode in self.photoNodes:
            pNode.__resetSpeciesList__(speciesList)

        #update all of the user nodes
        for uNode in self.userNodes:
            uNode.__resetSpeciesList__(speciesList)



    def __readin_gold__(self):
        print("Reading in expert classification")
        reader = csv.reader(open(self.baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
        next(reader, None)

        for row in reader:
            photoStr = row[2]
            species = row[12]

            photoIndex = self.photo_id_list.index(photoStr)
            photoNode = self.subjectNodes[photoIndex]

            photoNode.__updateGoldStandard__(species)



s = SnapshotSerengeti()
s.__readUserClassifications__()
s.__setSpeciesList__(["zebra"])

c = MajorityVote.MajorityVote(s.__getPhotoNodes__(),s.__getUserNodes__())
c.__classify__()
#s.__EM__()
#s.__readin_gold__()
#s.__gold_compare__()