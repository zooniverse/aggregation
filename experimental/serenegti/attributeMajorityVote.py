#!/usr/bin/env python
__author__ = 'greghines'
#load the stuff related to the Snapshot Serengeti
from SnapshotSerengeti import SnapshotSerengeti

#load the stuff related to the attribute based majority voting
import sys
sys.path.append("/Users/greghines/PycharmProjects/reduction/experimental/")
from graphicalClassification.attributeBased.majorityVote.UserNode import UserNode
from graphicalClassification.attributeBased.majorityVote.SubjectNode import SubjectNode
from graphicalClassification.attributeBased.majorityVote.MajorityVote import Classifier

import numpy as np

class SerengetiUser(UserNode):
    def __init__(self):
        UserNode.__init__(self)
        self.timeStamps = []

    def __addAnimal__(self,timeStamp,pNode,animal):
        if pNode in self.subjectsViewed:
            pIndex = self.subjectsViewed.index(pNode)
        else:
            self.subjectsViewed.append(pNode)
            self.timeStamps.append(None)
            self.classifications.append([])
            pIndex = len(self.subjectsViewed)-1

        self.timeStamps[pIndex] = timeStamp

        if animal != "":
            self.classifications[pIndex].append(animal)


    def __getTimeStamp__(self,pNode):
        return self.timeStamps[self.subjectsViewed.index(pNode)]

class SerengetiPhoto(SubjectNode):
    def __init__(self):
        SubjectNode.__init__(self)

    def __updateGoldStandard__(self,species):
        if self.goldStandard is None:
            self.goldStandard = [species]
        else:
            self.goldStandard.append(species)

s = SnapshotSerengeti(SerengetiUser,SerengetiPhoto)
s.__readUserClassifications__()
s.__readin_gold__()
#s.__setSpeciesList__(["zebra"])
t = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']

c = Classifier(s.__getPhotoNodes__(),s.__getUserNodes__())
#lx,ly = c.__analyze__(t)
#print "printing out"
import matplotlib.pyplot as plt

#plt.plot(lx,ly)
#lx.extend([0,1])
#ly.extend([0,0])

p = c.__alphaPlot__(t)
plt.plot(np.arange(0,1.01,0.02),p)

#plt.axis([0,1,0,1])
plt.show()

def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

#print area(zip(lx,ly))

#s.__EM__()
#s.__readin_gold__()
#s.__gold_compare__()