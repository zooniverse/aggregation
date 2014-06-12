__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np

import pymongo

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class PhotoNode:
    def __init__(self):
        self.classifiers = []

    def __add_classifier__(self,user):
        self.classifiers.append(user)

    #def __calc_likelihood__(self,speciesGroup):
    #    #calculate the likelihood of a particular speciesGroup - which may contain more than one species


class UserNode:
    def __init__(self,confusion_matrix= None,photos= [], classifications= []):
        self.photos = photos
        self.classifications = classifications

        #set up an initial confusion matrix - assume the user is perfect :)
        self.confusion_matrix = confusion_matrix


    def __set_species_grouping__(self,grouping):
        self.grouping = grouping
        p = powerset(grouping)

        #create the confusion matrix - start off by assuming that the user is perfect
        self.confusion_matrix = np.identity((len(p),len(p)))


    def __add_classification__(self,photo,classification):
        if photo in self.photos:
            photoIndex = self.photos.index(photo)
            self.classifications[photoIndex].append(classification)
        else:
            self.photos.append(photo)
            self.classifications.append([classification,])





class Classifier:
    def __init__(self):
        self.users = []
        self.photos = []

        self.client = pymongo.MongoClient()
        self.db = self.client['serengeti_2014-06-01']

    def __add_user__(self,user):
        self.users.append(user)

    def __add_photo__(self,photo):
        self.photos.append(photo)

    def __add_classification__(self,user,photo,classification):
        pass


class SerengetiClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)

    def __mongo_in__(self):
        collection = self.db['merged_classifications']

        for classification in collection.find():
            user_name= classification["user_name"]
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            timeStamp = classification["created_at"]
            annotations = classification["annotations"][0]

            #### - insert something to read in the time stamp of when the user tagged this photo
            #### - and then



