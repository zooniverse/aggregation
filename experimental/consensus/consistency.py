#!/usr/bin/env python
import os
import csv
import sys
import numpy as np
import math


if os.path.isdir("/Users/greghines/Databases/serengeti"):
    baseDir = "/Users/greghines/Databases/serengeti/"
else:
    baseDir = "/home/ggdhines/Databases/serengeti/"

classifications = {}
viewedSubjects = {}
individualConsistency = {}
subjectConsistency = {}
weights = {}

reader = csv.reader(open(baseDir+"goldFiltered.csv","rU"), delimiter=",")
next(reader, None)
for line in reader:
    subject_zooniverse_id = line[2]
    user_name = line[1]
    attribute = line[11]

    if not(subject_zooniverse_id in classifications):
        classifications[subject_zooniverse_id] = {}
        subjectConsistency[subject_zooniverse_id] = {}

    if not(user_name in classifications[subject_zooniverse_id]):
        if attribute is not "":
            classifications[subject_zooniverse_id][user_name] = [attribute]
        else:
            classifications[subject_zooniverse_id][user_name] = []

        subjectConsistency[subject_zooniverse_id][user_name] = 1
    else:
        if attribute is not "":
            classifications[subject_zooniverse_id][user_name].append(attribute)

    if not(user_name in viewedSubjects):
        viewedSubjects[user_name] = []
        individualConsistency[user_name] = 1.
        weights[user_name] = 1.

    if not(subject_zooniverse_id in viewedSubjects[user_name]):
        viewedSubjects[user_name].append(subject_zooniverse_id)

results = []
# numViewed = []
# for subject_zooniverse_id in classifications:
#     differentClassifications = []
#
#     #sum up the count for each classification
#     for user_name in classifications[subject_zooniverse_id]:
#         c = tuple(sorted(classifications[subject_zooniverse_id][user_name]))
#         if not(c in differentClassifications):
#             differentClassifications.append(c)
#
#     numViewed.append(len(differentClassifications))
#
# numClassifications = 0
# for user_name in viewedSubjects:
#     numClassifications += len(viewedSubjects[user_name])
#
# print numClassifications
#
# print np.mean(numViewed)
# print np.median(numViewed)
# xVal = range(1,max(numViewed))
# yVal = [len([i for i in numViewed if i == x])/float(len(numViewed)) for x in xVal]
# import matplotlib.pyplot as plt
# plt.bar([x-0.5 for x in xVal],yVal)
# plt.show()
# assert False

import pylab as P
wParam1 = 0.6
wParam2 = 8.5



for iterCount in range(2):
    #first time through
    for subject_zooniverse_id in classifications:
        classificationCount = {}
        totalClassifications = 0.
        uniqueClassifications = 0.
        #sum up the count for each classification
        for user_name in classifications[subject_zooniverse_id]:
            totalClassifications += 1
            c = tuple(sorted(classifications[subject_zooniverse_id][user_name]))
            w = weights[user_name]
            if not(c in classificationCount):
                classificationCount[c] = w
                uniqueClassifications += 1
            else:
                classificationCount[c] += w

        classificationPercentage = {c: classificationCount[c]/totalClassifications for c in classificationCount}
        #now calculate the consistency values
        for c in classificationCount:
            subjectConsistency[subject_zooniverse_id][c] = (classificationPercentage[c] + sum([1-classificationPercentage[cPrime] for cPrime in classificationCount if (c != cPrime)]))/uniqueClassifications
            if subjectConsistency[subject_zooniverse_id][c] < 0:
                print classificationPercentage
            assert(subjectConsistency[subject_zooniverse_id][c] >= 0)

    #calculate the user average
    for user_name in viewedSubjects:
        totalConsistency = 0
        for subject_zooniverse_id in viewedSubjects[user_name]:
            c = tuple(sorted(classifications[subject_zooniverse_id][user_name]))
            totalConsistency += subjectConsistency[subject_zooniverse_id][c]

        assert(totalConsistency >= 0)
        individualConsistency[user_name] = totalConsistency/float(len(viewedSubjects[user_name]))

    #map the consistency values into weights
    #weights = []
    # for user_name in individualConsistency:
    #     try:
    #         weights.append(min(1., math.pow(individualConsistency[user_name]/0.6,8.5)))
    #     except ValueError:
    #         print individualConsistency[user_name]
    #         raise
    weights = {user_name: min(1., math.pow(individualConsistency[user_name]/wParam1,2)) for user_name in individualConsistency}
    #results.append([individualConsistency[user_name] for user_name in individualConsistency])
    results.append([weights[user_name] for user_name in weights])

#print len(individualConsistency)
from scipy.stats import ks_2samp
print ks_2samp(results[0], results[1])


P.hist(results[0], 50, normed=1, histtype='step', cumulative=True)
P.show()