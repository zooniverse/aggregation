__author__ = 'greghines'
from copy import deepcopy
import numpy as np

class BaseUserNode:
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

class BaseSubjectNode:
    def __init__(self,numClasses=1):
        #self.nodeIndex = nodeIndex
        #self.name=name
        self.user_l = []
        self.mostlikely_classification = None

        self.num_classes = numClasses
        self.prior_probability = [1/float(self.num_classes) for i in range(self.num_classes)]
        self.updated = False
        self.correctly_classified = True

        self.oldLikelihood = None
        self.goldStandard = None



    def __getGoldStandard__(self):
        assert(self.goldStandard is not None)
        return self.goldStandard


    def __getNumClassifications__(self):
        return len(self.user_l)

    def __classifiedBy__(self,userNode):
        return userNode in self.user_l

    def __addUser__(self,user):
        self.user_l.append(user)

    def __update_priors__(self,new_priors):
        self.prior_probability = deepcopy(new_priors)

    def __get_mostlikely_classification__(self):
        assert(self.mostlikely_classification != None)
        return self.mostlikely_classification

    def __get_num_users__(self):
        return len(self.user_l)

    def __was_updated__(self):
        return self.updated

    def __getWeightedVote__(self,):
        for user in self.user_l:
            pass

    def __calc_mostlikely_classification__(self):
        #make sure at least one of the users has updated their confusion matrix
        updated = [u.__was_updated__() for u in self.user_l]
        if not(True in updated):
            self.updated = False
            return



        max_likelihood = -1

        old_classification = self.mostlikely_classification


        likelihood_list = [self.__get_class_likelihood__(class_index) for class_index in range(self.num_classes)]
        max_likelihood = max(likelihood_list)
        self.mostlikely_classification = likelihood_list.index(max_likelihood)

        if self.name == "ASG000f182":
            print("==")
            print(self.oldLikelihood)
            print(likelihood_list)
        self.oldLikelihood = likelihood_list[:]

        assert(self.mostlikely_classification != None)
        self.updated = not(self.mostlikely_classification == old_classification)




    def __get_class_likelihood__(self, class_index):
        #start off with the prior probability
        p = self.prior_probability[class_index]

        for user in self.user_l:
            p = p * user.__get_confusion_distribution__(self)[class_index]

        return p

    def __getVotes__(self,classList):
        votes = [0 for i in range(len(classList))]

        for user in self.user_l:
            votes[user.__getMostlikelyClassification__(self)] += 1

        totalVotes = sum(votes)
        return [v/float(totalVotes) for v in votes]


    #def __calc_likelihood__(self,speciesGroup):
    #    #calculate the likelihood of a particular speciesGroup - which may contain more than one species
