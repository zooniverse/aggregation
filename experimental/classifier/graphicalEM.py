#!/usr/bin/env python
__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np
import csv
import math
import pymongo
import operator
import os
from copy import deepcopy
import random

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class SubjectNode:
    def __init__(self,subjectName,debug = False):
        self.subjectName = subjectName
        self.users = []
        self.n = None
        self.classification = None

        #only for testing purposes
        self.goldStandard = None
        self.currentGold = None
        self.debug = debug

    def __getName__(self):
        return self.subjectName

    def __getNumUsers__(self):
        return len(self.users)

    def __addUser__(self,uNode):
        self.users.append(uNode)

    def __contains__(self):
        assert(self.classification is not None)
        return self.classification > 0

    def __correct__(self, classification=None):
        assert(self.currentGold is not None)

        if classification is not None:
            return classification == self.currentGold
        else:
            assert(self.classification is not None)
            return self.classification == self.currentGold




    def __setSpecies__(self,species):
        self.n = 2**len(species)
        powerSpecies = list(powerset(species))

        if self.goldStandard is not None:
            for i, required in enumerate(powerSpecies):
                prohibited = [s for s in species if not(s in required)]

                if not(set(required).intersection(self.goldStandard) == set(required)):
                    pass
                elif not(set(prohibited).intersection(self.goldStandard) == set()):
                    pass
                else:
                    self.currentGold = i
                    found = True
                    break

            assert found


    def __getClassification__(self):
        assert(self.classification is not None)
        return self.classification

    def __updateClassification__(self):
        #what is the probability of this particular set of species occurring?
        #
        multProbabilities = [reduce(operator.mul, [u.__getProbability__(self.subjectName,i) for u in self.users], 1) for i in range(self.n)]
        sumProbabilities = [sum([u.__getProbability__(self.subjectName,i) for u in self.users]) for i in range(self.n)]
        probabilities = multProbabilities
        if self.debug:
            print [u.__getReported__(self.subjectName) for u in self.users]
            print [u.__getProbability__(self.subjectName,0) for u in self.users]
            print [u.__getProbability__(self.subjectName,1) for u in self.users]
            print multProbabilities[0]
            print sumProbabilities[0]
            print "---"
            print multProbabilities[1]
            print sumProbabilities[1]
            print self.currentGold
            print "===="
        assert(max(probabilities) > 0)

        #if we have a tie, break it at random
        if probabilities[0] == probabilities[1]:
            self.classification = random.randint(0,1)
        else:
            self.classification = probabilities.index(max(probabilities))

    def __updateGoldStandard__(self,attribute):
        if self.goldStandard is None:
            self.goldStandard = [attribute]
        else:
            self.goldStandard.append(attribute)


class UserNode:
    def __init__(self,debug=False):
        self.classifications = {}
        self.subjects = {}
        self.species = None
        self.confusionMatrix = None

        self.reported = {}
        self.defaultConfusion = np.array([[0.75,0.25],[0.25,0.75]])
        self.debug = debug

    def __getNumSubjectsViewed__(self):
        return len(self.subjects)

    def __getClassifications__(self):
        classifications = []
        for subjectName in self.classifications:
            classifications.append((subjectName,self.reported[subjectName]))
        return classifications

    def __addSubject__(self,subjectName, sNode):
        if not(subjectName in self.subjects):
            self.subjects[subjectName] = sNode

    def __addAttribute__(self,subjectName,attribute):
        assert(subjectName in self.subjects)

        if not(subjectName in self.classifications):
            if attribute is "":
                self.classifications[subjectName] = []
            else:
                self.classifications[subjectName] = [attribute]
        elif attribute is not "":
            self.classifications[subjectName].append(attribute)

    def __setSpecies__(self,species):
        if len(species) == 1:
            self.confusionMatrix = deepcopy(self.defaultConfusion)
            self.n = 2**len(species)
        else:
            raise

        powerSpecies = list(powerset(species))
        for subjectName in self.classifications:
            c = self.classifications[subjectName]

            found = False

            for i, required in enumerate(powerSpecies):
                prohibited = [s for s in species if not(s in required)]

                if not(set(required).intersection(c) == set(required)):
                    pass
                elif not(set(prohibited).intersection(c) == set()):
                    pass
                else:
                    self.reported[subjectName] = i
                    found = True
                    break

            assert found

    def __getProbability__(self,subjectName,s):
        r = self.reported[subjectName]
        assert(type(r) == int)
        assert(type(s) == int)

        return self.confusionMatrix[r][s]

    def __getReported__(self,subjectName):
        return self.reported[subjectName]

    def __updateConfusionMatrix__(self):
        if self.debug:
            print "===---"
            print self.confusionMatrix
        self.confusionMatrix = np.zeros((self.n,self.n))
        for subjectName in self.subjects:
            sNode = self.subjects[subjectName]
            classification = sNode.__getClassification__()
            r = self.reported[subjectName]

            self.confusionMatrix[r][classification] += 1

        for i in range(self.n):
            t = sum(self.confusionMatrix[i])
            if t == 0:
                #revert to default
                self.confusionMatrix[i] = self.defaultConfusion[i][:]
            else:
                self.confusionMatrix[i] = self.confusionMatrix[i]/float(t)

        if self.debug:
            print self.confusionMatrix

        #print self.confusionMatrix

class graphicalEM:
    def __init__(self):

        self.userDict = {}
        self.subjectDict = {}


        self.gold_standard = {}

        if os.path.isdir("/Users/greghines/Databases/serengeti"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            self.baseDir = "/home/ggdhines/Databases/serengeti/"

    def __gold_compare__(self):
        print("comparing")
        correct = 0
        total = 0.
        for photoID in self.gold_standard:
            total += 1.
            expert_classification = self.gold_standard[photoID]

            #what class index would this count as?
            found = False

            for c_index, c in enumerate(powerset(self.speciesList)):
                c_complement = [s for s in self.speciesList if not(s in c)]

                if not(set(expert_classification).intersection(c) == set(c)):
                    pass
                elif not(set(expert_classification).intersection(c_complement) == set()):
                    pass
                else:
                    found = True
                    break

            if not found:
                print(expert_classification)
                print(self.species)
                print(self.class_l)
            assert(found)

            pNode = self.subjectNodes[self.photo_id_list.index(photoID)]
            mostlikely_classification = pNode.__get_mostlikely_classification__()

            if c_index == mostlikely_classification:
                correct += 1

        print(correct/total)



    def __readIndividualClassifications__(self):
        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rU"), delimiter=",")
        next(reader, None)
        for line in reader:
            subject_zooniverse_id = line[2]
            user_name = line[1]
            attribute = line[11]

            if not(subject_zooniverse_id in self.subjectDict):
                if len(self.subjectDict) < 10:
                    self.subjectDict[subject_zooniverse_id] = SubjectNode(subject_zooniverse_id)
                else:
                    self.subjectDict[subject_zooniverse_id] = SubjectNode(subject_zooniverse_id)
            sNode = self.subjectDict[subject_zooniverse_id]

            if sNode.__getNumUsers__() >= 10:
                continue

            if not(user_name in self.userDict):
                if len(self.userDict) < 20:
                    self.userDict[user_name] = UserNode()
                else:
                    self.userDict[user_name] = UserNode()
            uNode = self.userDict[user_name]

            #if this is not the first time this user has "tagged" this photo, nothing should happen
            sNode.__addUser__(uNode)
            uNode.__addSubject__(subject_zooniverse_id,sNode)
            uNode.__addAttribute__(subject_zooniverse_id,attribute)

    def __setSpecies__(self,speciesList):
        for subject in self.subjectDict:
            self.subjectDict[subject].__setSpecies__(speciesList)

        for user in self.userDict:
            self.userDict[user].__setSpecies__(speciesList)

    def __averageSubjectsViewed__(self):
        numUsers = 0.
        totalViews = 0

        for user_name in self.userDict:
            uNode = self.userDict[user_name]
            totalViews += uNode.__getNumSubjectsViewed__()
            numUsers += 1

        print "===---"
        print len(self.subjectDict)
        print numUsers
        print totalViews/numUsers

    def __updateClassifications__(self):
        for subject in self.subjectDict:
            self.subjectDict[subject].__updateClassification__()

        for user_name in self.userDict:
            self.userDict[user_name].__updateConfusionMatrix__()

    def __readGoldStandard__(self):
        reader = csv.reader(open(self.baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
        next(reader, None)

        for row in reader:
            photoStr = row[2]
            attribute = row[12]

            sNode = self.subjectDict[photoStr]
            sNode.__updateGoldStandard__(attribute)

    def __analyze__(self):
        total = 0.
        correct = 0.
        numPos = 0.
        correctPos = 0
        numNeg = 0.
        correctNeg = 0

        for subject_zooniverse_id in self.subjectDict:
            sNode = self.subjectDict[subject_zooniverse_id]

            if sNode.__correct__():
                correct += 1

            if sNode.__contains__():
                numPos += 1
                if sNode.__correct__():
                    correctPos += 1
            else:
                numNeg += 1
                if sNode.__correct__():
                    correctNeg += 1

            total += 1

        print correct/total
        print correctPos/numPos
        print correctNeg/numNeg

if __name__ == "__main__":
    c = graphicalEM()
    c.__readIndividualClassifications__()
    c.__readGoldStandard__()
    c.__setSpecies__(["zebra"])
    c.__updateClassifications__()
    c.__analyze__()