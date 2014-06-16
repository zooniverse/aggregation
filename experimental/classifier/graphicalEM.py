__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np
import math
import pymongo


def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class PhotoNode:
    def __init__(self):
        self.classifiers = []
        self.most_likely_class = None

    def __add_classifier__(self,user):
        self.classifiers.append(user)

    def __get_most_likely_group__(self):
        return self.most_likely_class



    def __get_maximum_likelihood__(self):
        max_likelihood = 0

        for group_index in range(len(self.group_l)):
            likelihood = self.__get_group_likelihood__(group_index)

            if likelihood > max_likelihood:
                max_likelihood = likelihood
                self.most_likely_class = group_index

    def __reset_species_groups__(self,species_l):
        self.most_likely_class = None
        

    def __get_group_likelihood__(self, group_index):
        #start off with the prior probability
        p = self.prior_probability[group_index]

        for user in self.classifiers:
            p = p * user.__get_confusion_distribution__(self)[group_index]

        return p


    #def __calc_likelihood__(self,speciesGroup):
    #    #calculate the likelihood of a particular speciesGroup - which may contain more than one species


class UserNode:
    def __init__(self):
        self.photos = []
        self.classifications = []
        self.mapped_classification = []

        #set up an initial confusion matrix - assume the user is perfect :)
        self.confusion_matrix = None
        self.grouping = None

    def __reset_species_groups__(self, species_l):
        self.mapped_classification = []

        species_groups = powerset(species_l)

        self.confusion_matrix = np.identity(len(species_groups))

        #now go through each of the photos the user has classified and see
        #which of the groups those classifications correspond to
        for c in self.classifications:
            for g_index, g in enumerate(species_groups):

                g_complement = [s for s in species_l if not(s in g)]
                if not(list(set(c).intersection(g)) == g):
                    pass
                elif not(list(set(c).intersection(g_complement)) == []):
                    pass
                else:
                    self.mapped_classification.append(g_index)
                    break

    def __get_confusion_distribution__(self,photo):
        photo_index = self.photos.index(photo)
        m_classification = self.mapped_classification[photo_index]

        return self.confusion_matrix[m_classification, :]

    def __update_confusion_matrix__(self):
        self.confusion_matrix = np.zeros(len(species_groups))

        for p, user_classification in zip(self.photos,self.mapped_classification):
            #what did that photo p decide to be classified as?
            actual_classification = p.__get_classification__()
            self.confusion_matrix[user_classification, actual_classification] += 1

        #normal so that each row sums to one
        for group_index in range(len(species_groups)):
            self.confusion_matrix[group_index, :] = 1/float(sum(self.confusion_matrix[group_index, :])) * self.confusion_matrix[group_index, :]




    def __add_classification__(self, photo, classification):
        assert(not(photo in self.photos))

        self.photos.append(photo)
        self.classifications.append(classification)


class Classifier:
    def __init__(self):
        self.users = []
        self.user_id_list = []

        self.photos = []
        #the following will help map from the zooniverse id to the photos
        self.photo_id_list = []
        #counts how many times each photo has been classified
        self.classification_count = []

        self.client = pymongo.MongoClient()
        self.db = self.client['serengeti_2014-06-01']

        self.cutoff = 10
        self.species_groups = list(powerset(["gazelleThomsons","gazelleGrants"]))

    def __mongo_in__(self):
        collection = self.db['merged_classifications']

        for classification in collection.find():
            user_name= classification["user_name"]
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            species_list = classification["species_list"]
            #for now - cheat :)
            #species_count = [1 for i in len(species_list)]

            if zooniverse_id in self.photo_id_list:
                photo = self.photos[self.photo_id_list.index(zooniverse_id)]
            else:
                self.photo_id_list.append(zooniverse_id)
                self.classification_count.append(0)
                self.photos.append(PhotoNode())
                photo = self.photos[-1]

            if photo.__get_num_classifications__() == self.cutoff:
                #if we have reached our limit
                continue

            #have we encountered this user before?
            if user_name in self.user_id_list:
                user = self.users[self.user_id_list.index(user_name)]
            else:
                self.users.append(UserNode())
                self.user_id_list.append(user_name)
                user = self.users[-1]

            user.__add_classification__(photo, species_list)
            photo.__add_classifier__(user)



