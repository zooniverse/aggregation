#!/usr/bin/env python
from experimental.old import IBCCsetup

__author__ = 'ggdhines'
import csv


class IBCCmajorityVote(IBCCsetup):
    def __init__(self,cuttoff=None):
        IBCCsetup.__init__(self,cuttoff)
        self.threshold = 0.5
        self.goldStandard = [[] for i in range(len(self.photoMappings))]

    def __mergeUserClassifications__(self):
        #classificationDict = dict.fromkeys(range(len(self.photoMappings)), [])
        classifications = [[] for i in range(len(self.photoMappings))]

        for species in self.speciesList:
            f = open(self.baseDir+"ibcc/"+species+"_ibcc.out"+str(self.cutOff),"rb")
            reader = csv.reader(f, delimiter=" ")

            for row in reader:
                pictureID = int(float(row[0]))
                userProb = float(row[2])

                if userProb > self.threshold:
                    classifications[pictureID].append(species)

        correct = 0
        for uC, eC in zip(classifications,self.goldStandard):
            if sorted(uC) == sorted(eC):
                correct += 1

        print correct/float(len(classifications))

    def __getExpertClassifications__(self):
        csvfile = open(self.baseDir+'/expert_classifications_raw.csv', 'rU')
        classificationReader = csv.reader(csvfile, delimiter=',')
        next(classificationReader, None)
        for row in classificationReader:
            photoID = row[2]
            photoIndex = self.photoMappings.index(photoID)
            species = row[12]

            if not (species in self.goldStandard[photoIndex]):
                self.goldStandard[photoIndex].append(species)

i = IBCCmajorityVote(10)
#i.__createConfigs__()
#i.__filterUserClassifications__()
#i.__ibcc__()
i.__getExpertClassifications__()
i.__mergeUserClassifications__()