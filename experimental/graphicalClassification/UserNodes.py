__author__ = 'greghines'
import numpy as np
from copy import deepcopy

class UserNode:
    def __init__(self,classList=None):
        #self.name = name
        self.subjectsViewed = []
        self.classifications = []

        if classList == None:
            self.numClasses = None
            self.confusion_matrix = None
        else:
            self.numClasses = 2**len(classList)

            #set up an initial confusion matrix - assume the user is perfect :)
            #self.class_l = list(powerset(species))
            self.confusion_matrix = np.identity(self.numClasses)
            self.confusion_matrix.fill(0.3)
            for i in range(self.numClasses):
                self.confusion_matrix[i][i] = 0.7

        self.updated = True


    def __reset_classes__(self, species_l):
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

    def __get_confusion_distribution__(self,subject):
        subjectIndex = self.subjectsViewed.index(subject)
        classification = self.classifications[subjectIndex]
        assert(classification != None)

        return self.confusion_matrix[classification, :]

    def __getMostlikelyClassification__(self,subject):
        subjectIndex = self.subjectsViewed.index(subject)
        classification = self.classifications[subjectIndex]

        return classification

    def __update_confusion_matrix__(self):
        #check to make sure that at least one of the classifications has changed
        #otherwise, skip
        updated = [p.__was_updated__() for p in self.subjectsViewed]
        if not(True in updated):
            self.updated = False
            return
        if self.name == "Gotalgebra":
            print(self.confusion_matrix)

        old_confusion_matrix = deepcopy(self.confusion_matrix)

        self.confusion_matrix = np.zeros((self.numClasses,self.numClasses))

        for subject, user_classification in zip(self.subjectsViewed,self.classifications):
            #what did that photo p decide to be classified as?
            actual_classification = subject.__get_mostlikely_classification__()

            self.confusion_matrix[user_classification, actual_classification] += 1

        #normal so that each row sums to one
        for group_index in range(self.numClasses):
            if sum(self.confusion_matrix[group_index, :]) == 0:
                t = np.zeros(self.numClasses)
                t.fill(1/float(self.numClasses))
                self.confusion_matrix[group_index,:] = t[:]
            else:
                self.confusion_matrix[group_index, :] = 1/float(sum(self.confusion_matrix[group_index, :])) * self.confusion_matrix[group_index, :]

        if self.name == "Gotalgebra":
            print(self.confusion_matrix)
        self.updated = False in (old_confusion_matrix == self.confusion_matrix)


    def __was_updated__(self):
        return self.updated

    def __add_classification__(self, photo, classification):
        if photo in self.photos:
            raise(PhotoAlreadyTagged())

        self.photos.append(photo)
        self.classifications.append(classification)

        found = False

        for c_index, c in enumerate(self.class_l):
            c_complement = [s for s in self.species_l if not(s in c)]

            if not(set(classification).intersection(c) == set(c)):
                pass
            elif not(set(classification).intersection(c_complement) == set([])):
                pass
            else:
                self.mapped_classification.append(c_index)
                found = True
                break

        if not found:
            print(classification)

        assert(found)
