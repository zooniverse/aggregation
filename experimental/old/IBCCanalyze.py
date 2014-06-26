#!/usr/bin/env python
from __future__ import print_function
from experimental.old import IBCCsetup

__author__ = 'greghines'
import csv
import cPickle as pickle
import urllib2
import json


class IBCCanalyse(IBCCsetup):
    def __init__(self,cutoff=None):
        IBCCsetup.__init__(self,cutoff)
        self.threshold = 0.5
        self.goldStandard = [[] for i in self.photoMappings]
        self.incorrectClassifications = []
        self.mistakeCount = {}

        self.classifications = None

    def __analyze__(self):
        self.__getExpertClassifications__()
        self.__getIncorrect__()
        self.__getMistakes__()

    def __getMetaData__(self):
        print(self.baseDir+"expert_classifications_raw.csv")
        classificationReader = csv.reader(self.baseDir+"expert_classifications_raw.csv","rU")
        next(classificationReader, None)
        for row in classificationReader:
            photoID = row[2]
            response = urllib2.urlopen('https://api.zooniverse.org/projects/serengeti/talk/subjects/'+photoID)
            obj = json.loads(response.read())
            print(obj["coords"])
            print(obj["created_at"])

            break




    def __getExpertClassifications__(self):
        try:
            f = open("/Users/greghines/Downloads/expert_classifications_raw.csv", 'rU')
        except IOError:
            f = open("/home/ggdhines/Databases/serengeti/expert_classifications_raw.csv", "rU")

        classificationReader = csv.reader(f, delimiter=',')
        next(classificationReader, None)
        for row in classificationReader:
            photoID = row[2]
            photoIndex = self.photoMappings.index(photoID)
            species = row[12]

            if not (species in self.goldStandard[photoIndex]):
                self.goldStandard[photoIndex].append(species)

    def __getMajorityPrediction__(self):
        self.classifications = [[] for i in range(len(self.photoMappings))]

        for species in self.speciesList:
            f = open(self.baseDir+"ibcc/"+species+"_ibcc.out"+str(self.cutOff),"rb")
            reader = csv.reader(f, delimiter=" ")

            for row in reader:
                pictureID = int(float(row[0]))
                userProb = float(row[2])

                if userProb > self.threshold:
                    self.classifications[pictureID].append(species)

    def __getPredictedMisclassifications__(self):
        self

    def __getIncorrect__(self):
        print("calculating incorrect predictions")

        correct = 0
        for index,(uC, eC) in enumerate(zip(self.classifications,self.goldStandard)):
            if sorted(uC) != sorted(eC):
                self.incorrectClassifications.append(self.photoMappings[index])
        print(len(self.incorrectClassifications))

    def __getMistakes__(self):
        print("getting individual mistakes")
        #get the list of all users we used in the simulation
        userList = pickle.load(open(self.baseDir+"/userList"+str(self.cutOff),"rb"))

        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rb"), delimiter=",")
        next(reader, None)
        mistakes = {}
        numClassifications = 0

        for row in reader:
            userStr = row[1]
            photoStr = row[2]
            speciesStr = row[11]

            #for now, only look at the specific users who went into our decision
            #if not(userStr in userList):
            #    continue

            if not(photoStr in self.incorrectClassifications):
                continue

            photoIndex = self.photoMappings.index(photoStr)
            correctClassification = self.goldStandard[photoIndex]

            #is this the first time we've encountered this user?
            if not(userStr in mistakes):
                mistakes[userStr] = {}
            user = mistakes[userStr]

            #is this the first time we've encountered this user AND photo
            if not(photoStr in user):
                #start by assuming that the user has not identified any species actually seen in the photo
                #but has also not "found" any species not actually in the photo

                user[photoIndex] = (tuple(correctClassification[:]),())
            numClassifications += 1
            if speciesStr != "":
                notFound = user[photoIndex][0]
                included = user[photoIndex][1]

                if speciesStr in correctClassification:
                    t = list(notFound)
                    t.remove(speciesStr)
                    notFound = tuple(t)
                else:
                    t = list(included)
                    t.append(speciesStr)
                    included = tuple(t)

                user[photoIndex] = tuple((notFound,included))
        print("printing out")
        print(numClassifications)
        for photoIndex in range(len(self.photoMappings)):
            for userStr in userList:
                if not(userStr in mistakes):
                    continue
                #did this user tag this photo?
                if photoIndex in mistakes[userStr]:
                    m = mistakes[userStr][photoIndex]
                    if (m != ([],[])) and (m[1] != ()):
                        if m in self.mistakeCount:
                            self.mistakeCount[m] += 1
                        else:
                            self.mistakeCount[m] = 1

        m = sorted(self.mistakeCount.keys(),key= lambda x:self.mistakeCount[x],reverse=True)
        for i in range(0,1):
            print(m[i],self.mistakeCount[m[i]])



f = IBCCanalyse()
f.__getMetaData__()