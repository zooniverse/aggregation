#!/usr/bin/env python
from __future__ import print_function
import csv
import os.path
import sys
if os.path.isdir("/Users/greghines/Code"):
    sys.path.append("/Users/greghines/Code/pyIBCC/python")
else:
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
import ibcc
import cPickle as pickle
__author__ = 'ggdhines'


class IBCCsetup:
    def __init__(self,cutoff=None):
        self.speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
        if cutoff == None:
            self.cutOff = 10
        else:
            self.cutOff = cutoff

        if os.path.isdir("/Users/greghines/Databases/serengeti"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            self.baseDir = "/home/ggdhines/Databases/serengeti/"

        self.photoMappings = []
        self.__createPhotoMappings()

    def __createConfigs__(self):
        print("making configuration files")
        for species in self.speciesList:
            #check to see whether or not this file exists
            if not(os.path.isfile(self.baseDir+"ibcc/"+str(species)+str(self.cutOff)+"config.py")):
                self.__createConfigFile(species,self.cutOff)

        print("done making configuration files")

    def __filterUserClassifications__(self):
        if not(os.path.isfile(self.baseDir+"goldFiltered.csv")):
            print("making gold standard")
            self._filterWithGoldStandard()

        for species in self.speciesList:
            print(species)
            #check to see whether or not this file exists
            if not(os.path.isfile(self.baseDir+"ibcc/"+species+"_ibcc.in"+str(self.cutOff))):
                self._filterWithSpecies(species,self.cutOff)




    def __createConfigFile(self,species,cutOff):
        f = open(self.baseDir+"ibcc/"+str(species)+str(self.cutOff)+"config.py",'wb')
        print("import numpy as np\nscores = np.array([0, 1])\nnClasses = 2\ninputFile =   '"+self.baseDir+"ibcc/"+species+"_ibcc.in"+str(cutOff)+"'",file=f)
        print("outputFile =  '"+self.baseDir+"ibcc/"+species+"_ibcc.out"+str(cutOff)+"'\nconfMatFile = '"+self.baseDir+"ibcc/"+species+"_ibcc.mat"+str(cutOff)+"'",file=f)
        f.close()

    def __createPhotoMappings(self):
        #pyIBCC requires every photo to be identified by an integer
        #use the photos listed in the expert classifications to create a mapping
        #so will have to change if we include photos with no corresponding expert classifications (but what won't)

        reader = csv.reader(open(self.baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
        next(reader, None)

        for row in reader:
            photoStr = row[2]
            if not(photoStr in self.photoMappings):
                self.photoMappings.append(photoStr)


    def _filterWithGoldStandard(self):
        #select only classifications for which we have a gold standard
        #have to rewrite a couple of things if we want to generalize
        goldStandardDict = {}
        photoCount = {}
        timeDict = {}


        f = open(self.baseDir+"goldFiltered.csv","wb")

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

        f.close()

    def _filterWithSpecies(self, requiredSpecies,cutOff,prohibitedSpecies=[],fID = None):
        #select entries only with regards to a specific species
        #output format in pyIBCC format
        #for now, assume that we have already filtered based on whether or not there exists a gold standard for that photo
        if type(requiredSpecies) != list:
            assert(type(requiredSpecies) == str)
            requiredSpecies = [requiredSpecies,]

        if prohibitedSpecies != []:
            assert(prohibitedSpecies == list)
            assert(prohibitedSpecies[0] == str)

        if fID == None:
            fID = requiredSpecies[0]

        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rb"), delimiter=",")

        #use indices in list to map from str to int
        userList = []
        photoList = []

        #userDict maps from userID to the list of all photos that user has tagged
        #photoDict maps form photos to the list of all users who have tagged that photo
        userDict = {}
        photoDict = {}

        next(reader, None)
        for row in reader:
            userStr = row[1]
            photoStr = row[2]
            speciesStr = row[11]

            #pyIBCC requires an integer id for each user, so convert the string into one
            #based on index of list, if not in list, add to list
            try:
                userID = userList.index(userStr)
                user = userDict[userID]
            except ValueError:
                userList.append(userStr)
                userID = len(userList) -1

                userDict[userID] = {}
                user = userDict[userID]



            photoID = self.photoMappings.index(photoStr)
            #is this the first time we've encountered this photo?
            if not(photoID in photoDict):
                photoDict[photoID] = []
            else:
                #since we assume that the cutoff is greater than 0 - only check this part if
                #this is not the first time we've seen this photo
                #have we already reached the max number of users for tagging this photo?
                if (len(photoDict[photoID]) >= cutOff) and not(userID in photoDict[photoID]):
                    continue

            if not(userID in photoDict[photoID]):
                photoDict[photoID].append(userID)

            #if this is the first time we've come across this user tagging this photo?
            if not(photoID in user):
                user[photoID] = ([False for i in requiredSpecies],[False for i in prohibitedSpecies])

            #is this species a required one?
            if speciesStr in requiredSpecies:
                user[photoID][0][requiredSpecies.index(speciesStr)] = True
            #else, is this species a prohibited one?
            elif speciesStr in prohibitedSpecies:
                userID[photoID][1][prohibitedSpecies.index(speciesStr)] = True


        fOut = open(self.baseDir+"ibcc/"+str(fID)+"_ibcc.in"+str(self.cutOff),"wb")
        #print out the user classifications
        for userID in userDict:
            user = userDict[userID]
            for photoID in user:
                classification = user[photoID]
                if not(False in classification[0]) and not(True in classification[1]):
                    f = 1
                else:
                    f = 0
                print(str(userID) + "," + str(photoID) + "," + str(f),file=fOut)
        fOut.close()

        #now write out the list of all users who have tagged a picture for which a gold standard exists
        pickle.dump(userList,open(self.baseDir+"/userList"+str(cutOff),"wb"))

    def __expertFilter__(self,desiredSpecies):
        #take the expert classifications and filter them for a specific species
        reader = csv.reader(open(self.baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
        next(reader, None)

        expertClassifications = [0 for i in range(len(self.photoMappings))]

        for row in reader:
            photoStr = row[2]
            speciesStr = row[12]

            photoID = self.photoMappings.index(photoStr)
            if speciesStr == desiredSpecies:
                expertClassifications[photoID] = 1

        #print out the expert classifications
        fOut = open(self.baseDir+desiredSpecies+".expert","wb")
        for photoID,classification in enumerate(expertClassifications):
            print(str(photoID) + "," + str(classification),file=fOut)
        fOut.close()

    def __ibcc__(self):
        for species in self.speciesList:
            #check to see whether or not this file exists
            if not(os.path.isfile(self.baseDir+"ibcc/"+species+"_ibcc.out"+str(self.cutOff))):
                ibcc.runIbcc(self.baseDir+"ibcc/"+str(species)+str(self.cutOff)+"config.py")



#i = IBCCsetup()
#i.__createConfigs__()
#i.__filterUserClassifications__()
#i.__ibcc__()