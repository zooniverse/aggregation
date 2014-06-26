#!/usr/bin/env python
from __future__ import print_function
__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np
import csv
import math
import pymongo
import os
from copy import deepcopy


class PhotoAlreadyTagged(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return "Photo was already tagged by user"







class Classifier:
    def __init__(self,subjectNodes,userNodes):
        self.userNodes = userNodes

        self.errorCount = 0
        self.subjectNodes = subjectNodes

        #the following will help map from the zooniverse id to the photos

        #counts how many times each photo has been classified
        #self.classification_count = []

        #self.client = pymongo.MongoClient()
        #self.db = self.client['serengeti_2014-06-01']

        #self.cutoff = 5
        #self.species = ["gazelleThomsons","gazelleGrants"]
        #self.class_l = list(powerset(self.species))

        self.gold_standard = {}

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
    c = Classifier()
    c.__readin_user__()
    c.__readin_gold__()
    c.__EM__()
    c.__gold_compare__()