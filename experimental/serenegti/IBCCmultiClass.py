__author__ = 'ggdhines'
from IBCCsetup import IBCCsetup
from __future__ import print_function
import csv
import os

class IBCCmultiClass(IBCCsetup):
    def __init__(self, cutoff):
        IBCCsetup.__init__(self, cutoff)

        self.species_groups = [["gazelleThomsons","gazelleGrants"],]
        species_in_groups = [item for sublist in self.species_groups for item in sublist]

        for species in self.speciesList:
            if not(species in species_in_groups):
                self.species_groups.append([species,])

    def _filterWithSpecies(self, speciesGroup, cutOff):
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



    def __filterUserClassifications__(self):
        if not(os.path.isfile("/home/ggdhines/Databases/serengeti/goldFiltered.csv")):
            self.__filterWithGoldStandard()

        for species in self.speciesList:
            #skip some species
            if species in self.speciesInMultiClass:
                continue

            #check to see whether or not this file exists
            if not(os.path.isfile("/home/ggdhines/Databases/serengeti/ibcc/"+species+"_ibcc.in"+str(self.cutOff))):
                self.__filterWithSpecies(species,self.cutOff)



    def _filterWithSpeciesClass(self, desiredSpecies, cutOff):
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
