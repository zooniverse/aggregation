#!/usr/bin/env python
__author__ = 'greghines'
from itertools import chain, combinations
import numpy as np
import csv
import math
import pymongo
import os
from copy import deepcopy


class Subject():
    def __init__(self,num_classes,gold=None):
        self.users = []
        self.num_classes = num_classes
        self.old_most_likely = None
        self.updated_most_likely = False
        self.prob_estimate = []
        self.gold = gold

        if gold != None:
            self.most_likely_class = gold
            self.prob_estimate = [0. for i in range(num_classes)]
            self.prob_estimate[gold] = 1.

    def __addUser__(self,user):
        self.users.append(user)

    def __calcProbability__(self):
        if self.gold != None:
            return

        if len(self.users) == 0:
            self.most_likely_class = 0
            self.updated_most_likely = False
            self.prob_estimate = [1,0]
        else:
            #print self.prob_estimate
            self.prob_estimate = []
            max_prob = 0
            self.most_likely_class = None

            # for user in self.users:
            #     r = user.__getReportedClassification__(self)
            #     confused = user.__getConfusion__()
            #     #print r
            #     #print confused[r]


            for class_index in range(self.num_classes):
                prob = 1
                for user in self.users:
                    prob = prob * user.__getProbability__(self,class_index)

                if prob > max_prob:
                    max_prob = prob
                    self.most_likely_class = class_index

                self.prob_estimate.append(prob)

            if self.most_likely_class is None:
                for class_index in range(self.num_classes):
                    print "***"
                    for user in self.users:
                        print  user.__getProbability__(self,class_index)

            assert(self.most_likely_class is not None)


            self.updated_most_likely = (self.most_likely_class != self.old_most_likely)
            self.old_most_likely = self.most_likely_class

            #print self.prob_estimate
            #self.prob_estimate = [p/sum(self.prob_estimate) for p in self.prob_estimate]
            self.prob_estimate = [0. for p in self.prob_estimate]
            self.prob_estimate[self.most_likely_class] = 1.
            #print self.prob_estimate
            assert max(self.prob_estimate) > 0
            #print self.prob_estimate
            #print "-"
            #print self.prob_estimate

    def __getMostLikely__(self):
        return self.most_likely_class

    def __updated__(self):
        return self.updated_most_likely

    def __getEstimate__(self):
        return self.prob_estimate

    def __getUsers__(self):
        return self.users

class User():
    def __init__(self,num_classes,confusion=None):
        self.classifications = {}
        self.num_classes = num_classes
        if confusion != None:
            self.confusion = confusion
        else:
            self.confusion = [[0.8,0.2],[0.2,0.8]]

        self.initial = deepcopy(confusion)
        self.count = [0. for i in range(self.num_classes)]

    def __addSubject__(self,subject,reported):
        self.classifications[subject] = reported
        self.count[reported] += 1

    def __getProbability__(self,subject,c):
        #how did the user classify this subject?
        reported_c = self.classifications[subject]
        #case where the user doesn't really have any input - so prob of 1 doesn't change anything

        assert(max(self.confusion[reported_c]) > 0)
        #if sum(self.count) < 25:
        #    return 1.
        return max(self.confusion[reported_c][c],0.001)

    def __updateConfusion__(self):
        if len(self.classifications) == 0:
            return

        self.confusion = [[0. for i in range(self.num_classes)] for j in range(self.num_classes)]

        assert len(self.classifications) > 0
        for subject in self.classifications.keys():
            #most_likely = subject.__getMostLikely__()
            estimates = subject.__getEstimate__()
            reported = self.classifications[subject]

            assert max(estimates) > 0

            

            for c in range(self.num_classes):
                self.confusion[reported][c] += estimates[c]

        assert max([max(c) for c in self.confusion]) > 0

        for class_index in range(self.num_classes):
            if max(self.confusion[class_index]) > 0:
                self.confusion[class_index] = [p/sum(self.confusion[class_index]) for p in self.confusion[class_index]]

        assert max([max(c) for c in self.confusion]) > 0

        #print self.confusion

    def __getClassification__(self,subject):
        return self.classifications[subject]

    def __getConfusion__(self):
        return [[round(c,3) for c in row] for row in self.confusion]

    def __getReportedClassification__(self,subject):
        return self.classifications[subject]




class IterativeEM():
    def __init__(self):
        self.user_list = []
        self.subject_list = []

    def __classify__(self,votes,num_classes,confusion=None,gold_values=None):
        for (userIndex,subjectIndex,v) in votes:
            while len(self.user_list) < (userIndex+1):
                current_index = len(self.user_list)
                #if type(confusion) == list:
                #    self.user_list.append(User(num_classes,confusion[current_index]))
                #else:
                self.user_list.append(User(num_classes,confusion))

            while len(self.subject_list) < (subjectIndex+1):
                current_index = len(self.subject_list)
                if (gold_values is not None) and (current_index in gold_values):
                    self.subject_list.append(Subject(num_classes,gold_values[current_index]))
                else:
                    self.subject_list.append(Subject(num_classes))

            user = self.user_list[userIndex]
            subject = self.subject_list[subjectIndex]

            user.__addSubject__(subject,v)
            subject.__addUser__(user)

        for i in range(50):
            print i
            for ii, subject in enumerate(self.subject_list):
                subject.__calcProbability__()

            any_updates = [subject.__updated__() for subject in self.subject_list]
            if not(True in any_updates):
                break

            for user in self.user_list:
                user.__updateConfusion__()



    def __getMostLikely__(self):
        return [subject.__getMostLikely__() for subject in self.subject_list]

    def __getEstimates__(self):
        return [subject.__getEstimate__() for subject in self.subject_list]

    def __getStats__(self,index):
        subject = self.subject_list[index]
        print subject.__getEstimate__()
        users = subject.__getUsers__()

        for u in users:
            print u.__getClassification__(subject)
            print u.__getConfusion__()
            print u.initial
            print u.count



