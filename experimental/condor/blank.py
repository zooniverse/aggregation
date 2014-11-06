#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import random

client = pymongo.MongoClient()
db = client['condor_2014-10-30']
condor_subjects = db["condor_subjects"]

image_annotations = []
reasons = set([])
blank_probabilities = []

blank_total = 0
consensus_blank_total = 0

# for subject in condor_subjects.find({"state":"complete"}):
#     try:
#         reason = subject["metadata"]["retire_reason"]
#         if reason == "blank":
#             blank_total += 1
#         elif reason == "blank_consensus":
#             consensus_blank_total += 1
#     except KeyError:
#         pass
#
# print blank_total
# print consensus_blank_total
#
# assert(False)
contains_condors = []

for subject in condor_subjects.find({"state":"complete"}):
    counter = 0
    annotations = []


    print subject
    print subject[u'location']["standard"]
    assert False
    try:

        reason = subject["metadata"]["retire_reason"]
        if reason in ["blank","blank_consensus"]:
            continue

        reasons.add(reason)

        for annotation_type in subject["metadata"]["counters"]:
            if annotation_type == "blank":
                annotations.extend([0 for i in range(subject["metadata"]["counters"][annotation_type])])
            else:
                counter += 1
                annotations.extend([counter for i in range(subject["metadata"]["counters"][annotation_type])])

        image_annotations.append(annotations[:])
        blank_probabilities.append(sum([1. for a in annotations if a == 0])/float(len(annotations)))

        condor_count = 0
        for annotation_type in subject["metadata"]["counters"]:
            if "condor" in annotation_type:
                condor_count += 1
                annotations.append(1)
            else:
                annotations.append(0)

        if condor_count >= 3:
            contains_condors.append(True)
        else:
            contains_condors.append(False)

    except KeyError:
        pass

print len(image_annotations)
#print reasons
first_N = 4
blanks = []
for i in range(100):
    blank_count = 0
    for annotation_distribution,cc in zip(image_annotations,contains_condors):
        users = np.random.choice(len(annotation_distribution),15)
        annotations = [annotation_distribution[u] for u in users]

        #would this have been classified as blank due to the first N initial users all saying blank
        if (annotations[:first_N] == [0 for i in range(first_N)]) and cc:
            blank_count += 1
        else:
            continue
            #find out which annotation, if any, was the first to achieve 5 "votes"
            first_to_5 = None
            lowest_index = float("inf")
            blank_annotations = [index for index,ann in enumerate(annotations) if ann == 0]


            for ann in range(1,max(annotations)):
                count = 0
                #count how many times this annotation appears - once we get to 5, stop
                for index,user_ann in enumerate(annotations):
                    if ann == 0:
                        print user_ann
                    if user_ann == ann:
                        count += 1

                    if count == 5:
                        break

                if (count == 5) and (index < lowest_index):
                    lowest_index = index
                    first_to_5 = ann


            #would we have retired by consensus at some point - if so we would have retired by blank
            #concensus first?
            #lowest_index is the lowest index that consensus was reached at for a non-blank classification
            blank_consensus_target = 8
            if (len(blank_annotations) >=blank_consensus_target) and (blank_annotations[blank_consensus_target-1] < lowest_index):
                blank_count += 1
                pass

    blanks.append(blank_count)

print np.mean(blanks)


counts = []
for i in range(200):
    false_blank_count = 0
    for blank_p in blank_probabilities:
        p = random.uniform(0,1)
        if p < (blank_p**first_N):
            false_blank_count += 1

    counts.append(false_blank_count)
    #print false_blank_count/float(len(blank_probabilities))

print np.mean(counts)
# plt.hist(counts,bins=20)
# plt.show()

# blankCount = {}
# alreadyRetired = []
# errorCount = 0
# active = 0
# nonBlankCount = 0
#
# import os
# animals = ["condor","eagle","turkeyVulture","raven","coyote"]
# for a in animals:
#     folder = "/home/greg/Databases/condors/images/"+a
#     for the_file in os.listdir(folder):
#         file_path = os.path.join(folder, the_file)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#         except Exception, e:
#             print e
#
# blankReasons = ["other-4","other-3","raven-2","carcassOrScale-1-raven-2","raven-12","carcassOrScale-4-raven-11","carcassOrScale-1-raven-11","carcassOrScale-1-raven-12","coyote-4","carcassOrScale-8","carcassOrScale-2-raven-17","carcassOrScale-3-raven-19","carcassOrScale-2-raven-19","raven-18","carcass-1-raven-20","raven-17","raven-21","carcassOrScale-4-raven-1","carcassOrScale-3-raven-1","carcass-2-other-1-raven-1","carcassOrScale-2-coyote-2","coyote-1-raven-1","carcassOrScale-1-coyote-2","coyote-2","raven-3","carcass-1-raven-4","carcass-2-raven-3","carcassOrScale-1-raven-7","raven-8","carcassOrScale-1-raven-9","carcassOrScale-1-raven-10","raven-7","carcass-1-raven-9","carcassOrScale-4-coyote-1","carcass-1-other-1","other-1","carcass-3-coyote-1","carcass-2-coyote-1","carcass-1-coyote-1","carcass-2-raven-13","carcass-2-raven-12","carcassOrScale-2-raven-11","carcassOrScale-2-raven-10","carcassOrScale-2-raven-12","carcass-2-raven-9","carcassOrScale-2-raven-9","carcassOrScale-2-raven-8","carcass-2-raven-2","carcass-1-raven-1","carcassOrScale-2-raven-2","carcassOrScale-1-coyote-1","carcassOrScale-1-coyote-1-raven-1","carcassOrScale-1-coyote-1","carcass-1-raven-6","carcass-2-raven-8","carcassOrScale-1-raven-6","raven-6","raven-5","carcassOrScale-2-raven-6","carcassOrScale-2-raven-4","carcass-1-raven-5","carcassOrScale-2-coyote-1","carcassOrScale-7","carcassOrScale-2-raven-15","carcassOrScale-2-raven-14","carcassOrScale-2-raven-13","raven-15","carcassOrScale-3-raven-15","carcassOrScale-3-raven-12","carcass-1-raven-15","carcassOrScale-4-raven-16","carcassOrScale-4-raven-14","carcassOrScale-1-raven-1","raven-1","carcassOrScale-6","carcass-5","coyote-1","blank","carcassOrScale-1","carcassOrScale-2","carcassOrScale-3","carcassOrScale-4","carcassOrScale-5","carcass-1","carcass-2","carcass-3","carcass-4","carcassOrScale-2-raven-1","carcass-4-coyote-1","carcass-3-raven-1","carcass-2-raven-1"]
#
# print len(blankReasons)
# for r in collection.find({"$and": [{"tutorial": False},{"subjects" : {"$elemMatch" : {"zooniverse_id" : {"$exists" : True}}}}]}):
#     try:
#         user_name = r["user_name"]
#     except KeyError:
#         continue
#
#     subject_id = r["subjects"][0]["zooniverse_id"]
#     _id = r["_id"]
#     if subject_id in alreadyRetired:
#         continue
#
#
#
#     if not(subject_id in blankCount):
#         blankCount[subject_id] = 0
#
#     if ("marks" in r["annotations"][-1]):
#         blank = 1
#         for markings in r["annotations"][-1]["marks"].values():
#
#             try:
#                 if markings["animal"] in ["condor","goldenEagle","turkeyVulture","coyote","raven"]:
#                     blank = 0
#                     break
#                 elif markings["animal"] in ["carcassOrScale","carcass","other"]:
#                     continue
#                 else:
#                     #print markings
#                     errorCount += 1
#             except KeyError:
#                 errorCount += 1
#     else:
#         blank = 1
#
#     blankCount[subject_id] += blank
#
#     if len(alreadyRetired) == 2000:
#         break
#
#     if blankCount[subject_id] == 3:
#         alreadyRetired.append(subject_id)
#         print len(alreadyRetired)
#
#         r2 = collection2.find_one({"zooniverse_id":subject_id})
#
#         try:
#             reason = r2["metadata"]["retire_reason"]
#             #print reason
#             if not(reason in ["blank","blank_consensus"]):
#                 #print r2["location"]["standard"]
#                 tagged = r2["metadata"]["counters"].keys()
#                 overlap = set()
#                 for t in tagged:
#                     for a in animals:
#                         if a in t:
#                             overlap.add(a)
#
#                 #print overlap
#                 #if ("condor" in t) or ("eagle" in t) or ("turkeyVulture" in t):
#                 if overlap != set():
#                     url =  r2["location"]["standard"]
#                     if not(os.path.isfile("/home/greg/Databases/condors/images/"+subject_id+".JPG")):
#                         urllib.urlretrieve(url, "/home/greg/Databases/condors/images/"+subject_id+".JPG")
#
#
#                     for a in overlap:
#                         print a
#                         os.symlink("/home/greg/Databases/condors/images/"+subject_id+".JPG","/home/greg/Databases/condors/images/"+a+"/"+subject_id+".JPG")
#
#         except KeyError:
#             #print " ++ " + r2["state"]
#             active += 1
#
# print len(alreadyRetired)
# print errorCount
# print active
# print nonBlankCount