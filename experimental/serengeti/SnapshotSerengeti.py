#!/usr/bin/env python
from __future__ import print_function
__author__ = 'greghines'
from itertools import chain, combinations
import csv
import math
import os
import sys

if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
else:
    sys.path.append("/Users/greghines/PycharmProjects/reduction/experimental/graphicalClassification")


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# class PhotoNode(SubjectNodes.SubjectNode):
#     def __init__(self):
#         SubjectNodes.SubjectNode.__init__(self)
#         self.expertTagged = []
#
#     def __updateExpertClassification__(self,tag):
#         self.expertTagged.append(tag)
#
#     def __resetSpeciesList__(self,speciesList):
#         self.most_likely_class = None
#         self.num_classes = int(math.pow(2,len(speciesList)))
#         self.prior_probability = [1/float(self.num_classes) for i in range(self.num_classes)]
#         self.updated = True
#
#         if self.expertTagged is not []:
#             self.goldStandard = None
#
#             for gIndex, group in enumerate(powerset(speciesList)):
#                 g_complement = [s for s in speciesList if not(s in group)]
#
#                 if not(set(self.expertTagged).intersection(group) == set(group)):
#                     pass
#                 elif not(set(self.expertTagged).intersection(g_complement) == set()):
#                     pass
#                 else:
#                     #self.mapped_classification.append(gIndex)
#                     self.goldStandard = gIndex
#                     break
#
#             assert(self.goldStandard is not None)




class SnapshotSerengeti:
    def __init__(self,container):
        #self.userNodes = []
        #self.photoNodes = []

        self.container = container
        self.timeStamps = {}

        self.limit = 10

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
            #if i >= 80000:
            #    break

            userName = row[1]
            photoName = row[2]
            timeStamp = row[4]
            tag = row[11]

            if tag == "":
                attributeList = []
            else:
                attributeList = [tag]

            #is this the first this photo has been tagged at all?
            if not(self.container.__subjectExists__(photoName)):
                self.container.__addSubject__(photoName)

            #is this the first time we encountered this user at all?
            if not(self.container.__userExists__(userName)):
                self.container.__addUser__(userName)


            #is this the first time this user has tagged this photo?
            #if not(pNode.__classifiedBy__(uNode)):
            if not(self.container.__classifiedBy__(photoName,userName)):
                #have we reached the limit for this particular photo?
                #if pNode.__getNumClassifications__() == self.limit:
                if self.container.__getNumClassifiers__(photoName) == self.limit:
                    #if so, skip this entry
                    continue
                else:
                    #otherwise add this user's classification
                    #photoDict[photoName][userName] = [timeStamp, newTags]
                    self.container.__newClassification__(photoName,userName,attributeList)
                    self.timeStamps[(photoName,userName)] = timeStamp
                    #pNode.__addUser__(uNode)
                    #uNode.__addAnimal__(timeStamp,pNode,tag)
            else:
                #have we reached the limit for this particular photo?
                if self.container.__getNumClassifiers__(photoName) == self.limit:
                    currentTimeStamp = self.timeStamps[(photoName,userName)]
                    #if a user tags multiple animals at once, they will be recorded as separate entries
                    #but HOPEFULLY with the same time stamp, so if the time stamps are the same
                    #add this entry
                    if timeStamp == currentTimeStamp:
                        currentClassification = self.container.__getClassification__(photoName,userName)
                        self.container.__updateAttributeList__(photoName,userName,attributeList)
                        #uNode.__addAnimal__(timeStamp,pNode,tag)
                    else:
                        continue
                else:
                    self.timeStamps[(photoName,userName)] = timeStamp
                    self.container.__updateAttributeList__(photoName,userName,attributeList)





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

            try:
                photoIndex = self.photo_id_list.index(photoStr)
            except ValueError:
                continue
            photoNode = self.photoNodes[photoIndex]

            photoNode.__updateGoldStandard__(species)



