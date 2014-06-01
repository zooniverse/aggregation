#!/usr/bin/env python
from __future__ import print_function
import csv
import os.path
import sys
sys.path.append("/home/ggdhines/github/pyIBCC/python")
import ibcc
__author__ = 'ggdhines'


class IBCCsetup:
    def __init__(self,cutoff=None):
        self.speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
        if cutoff == None:
            self.cutOff = 10
        else:
            self.cutOff = cutoff

        self.photoMappings = []
        self.__createPhotoMappings()

    def __createConfigs__(self):
        print("making configuration files")
        for species in self.speciesList:
            #check to see whether or not this file exists
            if not(os.path.isfile("/home/ggdhines/Databases/serengeti/ibcc/"+str(species)+str(self.cutOff)+"config.py")):
                self.__createConfigFile(species,self.cutOff)

    def __filterUserClassifications__(self):
        if not(os.path.isfile("/home/ggdhines/Databases/serengeti/goldFiltered.csv")):
            self.__filterWithGoldStandard()

        for species in self.speciesList:
            #check to see whether or not this file exists
            if not(os.path.isfile("/home/ggdhines/Databases/serengeti/ibcc/"+species+"_ibcc.in"+str(self.cutOff))):
                self.__filterWithSpecies(species,self.cutOff)




    def __createConfigFile(self,species,cutOff):
        f = open("/home/ggdhines/Databases/serengeti/ibcc/"+str(species)+str(self.cutOff)+"config.py",'wb')
        print("import numpy as np\nscores = np.array([0, 1])\nnClasses = 2\ninputFile =   '/home/ggdhines/Databases/serengeti/ibcc/"+species+"_ibcc.in"+str(cutOff)+"'",file=f)
        print("outputFile =  '/home/ggdhines/Databases/serengeti/ibcc/"+species+"_ibcc.out"+str(cutOff)+"'\nconfMatFile = '/home/ggdhines/Databases/serengeti/ibcc/"+species+"_ibcc.mat"+str(cutOff)+"'",file=f)
        f.close()

    def __createPhotoMappings(self):
        #pyIBCC requires every photo to be identified by an integer
        #use the photos listed in the expert classifications to create a mapping
        #so will have to change if we include photos with no corresponding expert classifications (but what won't)

        try:
            f = open("NA.csv", 'rb')
        except IOError:
            f = open("/home/ggdhines/Databases/serengeti/expert_classifications_raw.csv", "rU")
        reader = csv.reader(f, delimiter=",")
        next(reader, None)

        for row in reader:
            photoStr = row[2]
            if not(photoStr in self.photoMappings):
                self.photoMappings.append(photoStr)


    def __filterWithGoldStandard(self):
        #select only classifications for which we have a gold standard
        #have to rewrite a couple of things if we want to generalize
        goldStandardDict = {}
        photoCount = {}
        timeDict = {}

        f = open("/home/ggdhines/Databases/serengeti/goldFiltered.csv","wb")

        # #try to open the file first on my laptop, then try on my desktop
        # try:
        #     csvfile = open('/Users/greghines/Databases/expert_classifications.csv', 'rb')
        #     classificationReader = csv.reader(csvfile, delimiter=',')
        #     next(classificationReader,None)
        #
        # except IOError:
        #     csvfile = open('/home/ggdhines/Databases/serengeti/expert_classifications_raw.csv', 'rU')
        #     classificationReader = csv.reader(csvfile, delimiter=',')
        #
        # for row in classificationReader:
        #     photoID = row[2]
        #     species = row[12]
        #
        #     if photoID in goldStandardDict:
        #         if not (species in goldStandardDict[photoID]):
        #             goldStandardDict[photoID].append(species)
        #     else:
        #         goldStandardDict[photoID] = [species]
        #         photoCount[photoID] = []
        #         timeDict[photoID] = []

        #now go through the actual classifications
        #again, go try to open on my laptop first
        try:
            csvfile = open('/Users/greghines/Downloads/2014-05-18_serengeti_classifications.csv', 'rb')
        except IOError:
            csvfile = open('/home/ggdhines/Databases/serengeti/2014-05-25_serengeti_classifications.csv','rb')

        print(csvfile.readline()[:-1], file=f)
        #classificationReader = csv.reader(csvfile,delimiter=',')
        while True:
            line = csvfile.readline()
            if not line: break
            row = line.split(",")
            photoID = row[2][1:-1]
            if photoID in self.photoMappings:
                print(line[:-1], file=f)

    def _filterWithSpecies(self, desiredSpecies,cutOff):
        #select entries only with regards to a specific species
        #output format in pyIBCC format
        #for now, assume that we have already filtered based on whether or not there exists a gold standard for that photo
        try:
            f = open("NA.csv", 'rb')
        except IOError:
            f = open("/home/ggdhines/Databases/serengeti/goldFiltered.csv", "rb")

        reader = csv.reader(f, delimiter=",")

        userList = []
        photoList = []

        userDict = {}
        classifications = {}

        next(reader, None)
        for row in reader:
            userStr = row[1]
            photoStr = row[2]
            speciesStr = row[11]

            #pyIBCC requires an integer id for each user, so convert the string into one
            #based on index of list, if not in list, add to list
            try:
                userID = userList.index(userStr)
            except ValueError:
                userList.append(userStr)
                userID = len(userList) -1

            #userDict keeps track of all the photos this user has classified
            #user keeps track of the individual classifications
            try:
                user = userDict[userID]
            except KeyError:
                userDict[userID] = {}
                user = userDict[userID]

            photoID = self.photoMappings.index(photoStr)
            #if this is the first time we've come across this photo?
            if not(photoID in classifications):
                classifications[photoID] = [userID]
            else:
                c = classifications[photoID]
                #if this is the first time we've come across this particular user classifying this photo
                if not(userID in c):
                    #we've reached the desired max number of users for this photo
                    if len(c) > cutOff:
                        continue
                    else:
                        c.append(userID)

            # #check to see if this photo has been tagged at all before
            # try:
            #     photoID = photoList.index(photoStr)
            #     #if so, get the list of all the users who have tagged (or classified) this photo
            #
            #
            #
            #
            # except ValueError:
            #     photoList.append(photoStr)
            #     photoID = len(photoList) - 1
            #     classifications[photoID] = [userID]

            #has the user already tagged this photo?
            if photoID in user:
                user[photoID] = (speciesStr == desiredSpecies) or user[photoID]
            else:
                user[photoID] = (speciesStr == desiredSpecies)

        fOut = open("/home/ggdhines/Databases/serengeti/ibcc/"+desiredSpecies+"_ibcc.in"+str(self.cutOff),"wb")
        #print out the user classifications
        for photoID in classifications:
            for userID in classifications[photoID]:
                user = userDict[userID]
                if user[photoID] is True:
                    f = 1
                else:
                    f = 0
                print(str(userID) + "," + str(photoID) + "," + str(f),file=fOut)
        fOut.close()

    def __expertFilter__(self,desiredSpecies):
        #take the expert classifications and filter them for a specific species

        try:
            f = open("NA.csv", 'rb')
        except IOError:
            f = open("/home/ggdhines/Databases/serengeti/expert_classifications_raw.csv", "rU")
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        expertClassifications = [0 for i in range(len(self.photoMappings))]

        for row in reader:
            photoStr = row[2]
            speciesStr = row[12]

            photoID = self.photoMappings.index(photoStr)
            if speciesStr == desiredSpecies:
                expertClassifications[photoID] = 1

        #print out the expert classifications
        fOut = open("/home/ggdhines/Databases/serengeti/ibcc/"+desiredSpecies+"_ibcc.expert","wb")
        for photoID,classification in enumerate(expertClassifications):
            print(str(photoID) + "," + str(classification),file=fOut)
        fOut.close()

    def __ibcc__(self):
        for species in self.speciesList:
            #check to see whether or not this file exists
            if not(os.path.isfile("/home/ggdhines/Databases/serengeti/ibcc/"+species+"_ibcc.out"+str(self.cutOff))):
                ibcc.runIbcc("/home/ggdhines/Databases/serengeti/ibcc/"+str(species)+str(self.cutOff)+"config.py")



#i = IBCCsetup()
#i.__createConfigs__()
#i.__filterUserClassifications__()
#i.__ibcc__()