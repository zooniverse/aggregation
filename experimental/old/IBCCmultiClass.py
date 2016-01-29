#!/usr/bin/env python
from __future__ import print_function
import csv
import os
import cPickle as pickle
from itertools import chain, combinations

from experimental.old import IBCCsetup


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def subsets(s):
    return map(set, powerset(s))


class IBCCmultiClass(IBCCsetup):
    def __init__(self, cutoff):
        IBCCsetup.__init__(self, cutoff)

        self.species_groups = [["gazelleThomsons","gazelleGrants"],]
        #species_in_groups = [item for sublist in self.species_groups for item in sublist]
        #
        #for species in self.speciesList:
        #    if not(species in species_in_groups):
        #        self.species_groups.append([species,])



    def __filterUserClassifications__(self):
        if not(os.path.isfile(self.baseDir+"goldFiltered.csv")):
            self._filterWithGoldStandard()

        #check to see if we are already using the existing species group list
        #for my sanity, only one species group list is used at a time - will overwrite anything prexisting
        try:
            existingSpeciesGroups = pickle.load(open(self.baseDir+"speciesGroups.pickle","rb"))
            if existingSpeciesGroups == self.species_groups:
                #alreadyExisting = True
                return
            else:
                alreadyExisting = False
        except IOError:
            alreadyExisting = False

        pickle.dump(self.species_groups,open(self.baseDir+"speciesGroups.pickle","wb"))

        i= -1
        for group in self.species_groups:
            i+= 1
            powerSubset = subsets(group)
            j = -1
            for requiredSpecies in powerSubset:
                j+= 1
                if requiredSpecies == []:
                    continue

                prohibitedSpecies = [s for s in group if not(s in requiredSpecies)]
                self._filterWithSpecies(self, requiredSpecies,self.cutOff,prohibitedSpecies=prohibitedSpecies,fID = "g"+str(i)+"_"+str(j))

        species_in_groups = [item for sublist in self.species_groups for item in sublist]


        if alreadyExisting == False:


            for speciesGroup in self.species_groups:
                self.__filterWithSpecies(speciesGroup,self.cutOff)

    def _filterWithSpeciesGroup(self, speciesGroup, cutOff):
        #select entries only with regards to a specific species
        #output format in pyIBCC format
        #for now, assume that we have already filtered based on whether or not there exists a gold standard for that photo
        reader = csv.reader(open(self.baseDir+"goldFiltered.csv", "rb"), delimiter=",")

        userList = []
        userDict = {}

        next(reader, None)
        for row in reader:
            userStr = row[1]
            photoStr = row[2]
            speciesStr = row[11]
            assert(speciesStr[0] != "\"")

            #pyIBCC requires an integer id for each user, so convert the string into one
            #based on index of list, if not in list, add to list
            #userDict keeps track of all the photos this user has classified
            #user keeps track of the individual classifications
            try:
                userID = userList.index(userStr)
                user = userDict[userID]
            except ValueError:
                userList.append(userStr)
                userID = len(userList) -1

                userDict[userID] = {}
                user = userDict[userID]


            photoID = self.photoMappings.index(photoStr)
            #if this is the first time we've come across this photo?
            if not(photoID in user):
                user[photoID] = [False for i in range(len(speciesGroup))]

            try:
                #is the species the user found in the species group?
                sIndex = speciesGroup.index(speciesStr)
            except ValueError:
                continue

            user[photoID][sIndex] = True



        fOut = open("/home/ggdhines/Databases/serengeti/ibcc/"+desiredSpecies+"_ibcc.in"+str(self.cutOff),"wb")
        #print out the user classifications
        for userID in userDict:
            for photoID in userDict[userID]:
                user = userDict[userID]
                if not (False in user[photoID]):
                    f = 1
                else:
                    f = 0
                print(str(userID) + "," + str(photoID) + "," + str(f),file=fOut)
        fOut.close()

f = IBCCmultiClass(10)
f.__filterUserClassifications__()