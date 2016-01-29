#!/usr/bin/env python
from __future__ import print_function
import csv
import os
__author__ = 'greghines'
import pymongo
from itertools import chain, combinations


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class IBCCmongo:
    def __init__(self):
        if os.path.isdir("/Users/greghines/Databases/serengeti"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            self.baseDir = "/home/ggdhines/Databases/serengeti/"

        self.client = pymongo.MongoClient()
        self.db = self.client['serengeti_2014-06-01']

        self.goldClassified = []
        self.__createPhotoMappings()

        self.species_groups = [["gazelleThomsons","gazelleGrants"],]
        self.speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
        t = [item for sublist in self.species_groups for item in sublist]
        self.speciesList = [s for s in self.speciesList if not(s in t)]

        self.classification_l = []

    def __writeOut__(self):
        mCollections = self.db['merged_classifications']

        self.classification_l = []

        for zooniverse_id in self.goldClassified:
            for u in mCollections.find({"zooniverse_id":zooniverse_id})[0:10]:
                self.classification_l.append(u["_id"])

    def __IBCCgroup__(self):
        mCollections = self.db['merged_classifications']
        i = 0
        for group in self.species_groups:

            for required_species in list(powerset(group)):
                if required_species == ():
                    continue

                prohibited_species = [s for s in group if not(s in required_species)]
                print((required_species,prohibited_species))

                for userIndex,uID in enumerate(self.classification_l):
                    for classification in mCollections.find({"_id":uID}):

                        subjectID = classification["zooniverse_id"]
                        subjectIndex = self.goldClassified.index(subjectID)
                        speciesTagged = classification["species"]

                        #did they forget to tag any of the required species
                        if not(list(set(speciesTagged).intersection(required_species)) == required_species):
                            f = 0
                        #did they tag something they weren't supposed to?
                        elif not(list(set(speciesTagged).intersection(prohibited_species)) == []):
                            f = 0
                        else:
                            f = 1


                        if (('gazelleThomsons' in speciesTagged) or ('gazelleGrants' in speciesTagged)):
                            print(speciesTagged)
                            print(required_species)
                            print(set(speciesTagged).intersection(required_species))
                            print(tuple(set(speciesTagged).intersection(required_species)) == required_species)
                            print(tuple(set(speciesTagged).intersection(prohibited_species)) == ())
                            print((userIndex,subjectIndex,f))
                            print("===---")

                    i += 1
                    if i == 200:
                        return


    def __groupIterate__(self):
        for group in self.species_groups:
            required_species = list(powerset(group))
            if required_species == ():
                continue

            prohibited_species = [s for s in group if not(s in required_species)]

    def __createPhotoMappings(self):
        #pyIBCC requires every photo to be identified by an integer
        #use the photos listed in the expert classifications to create a mapping
        #so will have to change if we include photos with no corresponding expert classifications (but what won't)

        reader = csv.reader(open(self.baseDir+"expert_classifications_raw.csv", "rU"), delimiter=",")
        next(reader, None)

        for row in reader:
            photoStr = row[2]
            if not(photoStr in self.goldClassified):
                self.goldClassified.append(photoStr)

    def __mergeClassifications__(self):
        #since users may tag a photo as containing multiple animals
        #collect all similar ones into one record
        collection = self.db['filtered_classifications']
        mCollections = self.db['merged_classifications']


        currName = None
        currID = None
        currTime = None
        currSpecies = []
        currCount = []

        for classification in collection.find():
            user_name= classification["user_name"]
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            timeStamp = classification["created_at"]
            annotations = classification["annotations"][0]


            if (user_name != currName) and (zooniverse_id != currID):
                if currName != None:
                    r = {"user_name":currName,"zooniverse_id":currID,"created_at":currTime,"species":currSpecies,"count":currCount}
                    mCollections.insert(r)

                currID = zooniverse_id
                currName = user_name
                currTime = timeStamp
                currSpecies = []
                currCount = []

            if not("nothing" in annotations):
                currSpecies.append(annotations["species"])
                currCount.append(annotations["count"])

        r = {"user_name":currName,"zooniverse_id":currID,"created_at":currTime,"species":currSpecies,"count":currCount}
        mCollections.insert(r)



    def __csv_to_mongo__(self):
        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rb"), delimiter=",")
        filteredCollection = self.db['filtered_classifications']

        next(reader, None)
        for row in reader:
            classification = {}
            classification["user_name"] = row[1]
            if row[11] != "":
                try:
                    classification["annotations"] = [{"species":row[11], "count":int(row[12])}]
                except ValueError:
                    classification["annotations"] = [{"species":row[11], "count":-1}]
            else:
                classification["annotations"] = [{"nothing":"true"}]
            classification["subjects"] = [{"zooniverse_id":row[2]}]
            classification["created_at"] = row[4]
            filteredCollection.insert(classification)

    def __gold_filter__(self):
        collection = self.db['serengeti_classifications']
        filteredCollection = self.db['filtered_classifications']

        i = 0

        for classification in collection.find({'tutorial': {'$exists': False}}):
            zooniverse_id = str(classification["subjects"][0]["zooniverse_id"])
            break
            if zooniverse_id in self.goldClassified:
                i += 1
                print(i)
                filteredCollection.insert(classification)

f = IBCCmongo()
f.__writeOut__()
print("almost there")
f.__IBCCgroup__()