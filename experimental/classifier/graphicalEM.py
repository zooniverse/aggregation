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


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class SubjectNode:
    def __init__(self,subjectName):
        self.subjectName = subjectName
        self.users = []
        self.n = None
        self.classification = None

    def __addUser__(self,uNode):
        self.users.append(uNode)

    def __setSpecies__(self,species):
        self.n = 2**len(species)

    def __getClassification__(self):
        assert(self.classification is not None)
        return self.classification

    def __updateClassification__(self):
        #what is the probability of this particular set of species occurring?
        probabilities = [reduce(operator.mul, [u.__getProbability__(self.subjectName,i) for u in self.users], 1) for i in range(self.n)]
        self.classification = probabilities.index(max(probabilities))


class UserNode:
    def __init__(self):
        self.classifications = {}
        self.subjects = {}
        self.species = None
        self.confusionMatrix = None

        self.reported = {}

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
            self.confusionMatrix = np.array([[0.75,0.25],[0.25,0.75]])
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

    def __updateConfusionMatrix__(self):
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
                if i == 0:
                    self.confusionMatrix[i] = [0.75,0.25]
                else:
                    self.confusionMatrix[i] = [0.25,0.75]
            else:
                self.confusionMatrix[i] = self.confusionMatrix[i]/float(t)
        print self.confusionMatrix

class graphicalEM:
    def __init__(self):

        self.userDict = {}
        self.subjectDict = {}

        #the following will help map from the zooniverse id to the photos

        #counts how many times each photo has been classified
        #self.classification_count = []

        #self.client = pymongo.MongoClient()
        #self.db = self.client['serengeti_2014-06-01']

        #self.cutoff = 5
        #self.species = ["gazelleThomsons","gazelleGrants"]
        #self.class_l = list(powerset(self.species))

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

    def __classify__(self):
        for iter in range(2):
            print("running EM")
            updateCount = 0
            totalCount = 0.


            for subject in self.subjectNodes:
                subject.__calc_mostlikely_classification__()
                totalCount += 1.
                if subject.__was_updated__():
                    updateCount += 1

                #update the priors
                subject.__get_mostlikely_classification__()

            #print(counts)
            #break
            #for photo in self.photos:
            #    photo.__update_priors__(counts)



            print("update percentage: " + str(updateCount/totalCount))

            for user in self.userNodes:
                user.__update_confusion_matrix__()

    def __readIndividualClassifications__(self):
        reader = csv.reader(open(self.baseDir+"goldFiltered.csv","rU"), delimiter=",")
        next(reader, None)
        for line in reader:
            subject_zooniverse_id = line[2]
            user_name = line[1]
            attribute = line[11]

            if not(subject_zooniverse_id in self.subjectDict):
                self.subjectDict[subject_zooniverse_id] = SubjectNode(subject_zooniverse_id)
            sNode = self.subjectDict[subject_zooniverse_id]

            if not(user_name in self.userDict):
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

    def __updateClassifications__(self):
        for subject in self.subjectDict:
            self.subjectDict[subject].__updateClassification__()

        for user_name in self.userDict:
            self.userDict[user_name].__updateConfusionMatrix__()


    def __readin_user__(self):
        collection = self.db['merged_classifications'+str(self.cutoff)]
        print("Reading in mongodb collection")

        for classification in collection.find():
            user_name= classification["user_name"]
            zooniverse_id = classification["zooniverse_id"]
            species_list = classification["species"]
            #for now - cheat :)
            #species_count = [1 for i in len(species_list)]

            if zooniverse_id in self.photo_id_list:
                photo = self.photos[self.photo_id_list.index(zooniverse_id)]
            else:
                self.photo_id_list.append(zooniverse_id)
                self.photos.append(PhotoNode(self.species))
                photo = self.photos[-1]

            if photo.__get_num_users__() == self.cutoff:
                #if we have reached our limit
                continue

            #have we encountered this user before?
            if user_name in self.user_id_list:
                user = self.users[self.user_id_list.index(user_name)]
            else:
                self.users.append(UserNode(self.species))
                self.user_id_list.append(user_name)
                user = self.users[-1]

            try:
                user.__add_classification__(photo, species_list)
                photo.__add_user__(user)
            except PhotoAlreadyTagged:
                print((user_name,zooniverse_id))
                self.errorCount += 1

        print("double instances: " + str(self.errorCount))





if __name__ == "__main__":
    c = graphicalEM()
    c.__readIndividualClassifications__()
    c.__setSpecies__(["buffalo"])
    c.__updateClassifications__()