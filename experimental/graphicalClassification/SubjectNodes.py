from copy import deepcopy

__author__ = 'greghines'
class SubjectNode:
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

    def __getVotes__(self):
        votes = [0 for i in range(self.num_classes)]

        for user in self.user_l:
            votes[user.__getMostlikelyClassification__(self)] += 1

        totalVotes = sum(votes)
        return [v/float(totalVotes) for v in votes]


    #def __calc_likelihood__(self,speciesGroup):
    #    #calculate the likelihood of a particular speciesGroup - which may contain more than one species

