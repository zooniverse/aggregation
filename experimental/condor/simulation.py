#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import random
import cPickle as pickle

client = pymongo.MongoClient()
db = client['condor_2014-10-30']
condor_annotations = db["condor_classifications"]
condor_subjects = db["condor_subjects"]
overall_annotations = {}

false_blank = 0
completed = 0
completed_blank = 0
completed_consensus_blank =0
unknown =0
completed_set = set()
problem = 0
regular_retire = 0
false_completed_blank = 0

how_it_ended = pickle.load(open("/home/greg/condor_temp","r"))

cblank_to_blank = 0

#previously_completed = pickle.load(open("/home/greg/condor_temp","r"))

for i, session in enumerate(condor_annotations.find()):
    if session["subjects"] == []:
        continue

    zooniverse_id = session["subjects"][0]["zooniverse_id"]

    if zooniverse_id in completed_set:
        continue

    if len(session["annotations"]) < 4:
        if not(zooniverse_id in overall_annotations):
            overall_annotations[zooniverse_id] = {(("blank",1),):1}
        elif not((("blank",1),) in overall_annotations[zooniverse_id]):
            overall_annotations[zooniverse_id][(("blank",1),)] = 1
        else:
            overall_annotations[zooniverse_id][(("blank",1),)] += 1


    else:
        animal_count = {}
        for annotation_dict in session["annotations"][3]["marks"].values():
            try:
                animal = annotation_dict["animal"]
            except KeyError:
                #print annotation_dict
                continue

            # if animal == "carcassOrScale":
            #     #so that carcassOrScale counts as blank
            #     animal = "blank"

            if not(animal in animal_count):
                animal_count[animal] = 1
            else:
                animal_count[animal] += 1

        user_animals = animal_count.keys()
        if False: #user_animals == ["carcassOrScale"]:
            user_annotations = (("blank",1),)
        else:
            user_annotations = tuple(sorted(animal_count.items(),key= lambda x:x[0]))


        if user_annotations != ():
            #since we have skipped over any carcassOrScale
            #user_annotations might be empty, so only update if not empty
            if not(zooniverse_id in overall_annotations):
                overall_annotations[zooniverse_id] = {user_annotations:1}
            elif not(user_annotations in overall_annotations[zooniverse_id]):
                overall_annotations[zooniverse_id][user_annotations] = 1
            else:
                overall_annotations[zooniverse_id][user_annotations] += 1


    if "tutorial" in condor_subjects.find_one({"zooniverse_id":zooniverse_id}).keys():
        continue

    if not(zooniverse_id in overall_annotations):
        continue

    if overall_annotations[zooniverse_id] == {(("blank",1),):3}:
        #is this a false blank?
        completed_set.add(zooniverse_id)
        how_it_ended[zooniverse_id] = "blank"

        subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
        state = subject["state"]

        #record
        completed += 1
        completed_blank += 1

        if state == "complete":
            actual_reason = subject["metadata"]["retire_reason"]
            if not(actual_reason in ["blank","blank_consensus"]):
                false_blank += 1
                #if actual_reason == "no_condors_present":
                #    print subject

            if how_it_ended[zooniverse_id] == "blank_consensus":
                cblank_to_blank += 1


        else:
            #if not(zooniverse_id in previously_completed):
            #    print zooniverse_id
            unknown += 1
    elif max(overall_annotations[zooniverse_id].values()) >= 4:


        non_blank_reason = [ann for ann,votes in overall_annotations[zooniverse_id].items() if (votes >= 4) and (ann != (("blank",1),))]
        blank_reason = [ann for ann,votes in overall_annotations[zooniverse_id].items() if (votes >= 5) and (ann == (("blank",1),))]

        # print overall_annotations[zooniverse_id].items()
        # for ann in overall_annotations[zooniverse_id].keys():
        #     print ann == (("blank",1),)

        if (non_blank_reason != []) or (blank_reason != []):
            #we will retire for one reason for another
            completed_set.add(zooniverse_id)
            completed += 1
            subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})

            if non_blank_reason != []:
                #we retired it in the simulation because we thought it was non-blank
                regular_retire += 1
                how_it_ended[zooniverse_id] = "consensus"
                #what was the actual reason the image was retired - if it was retired at all
                state = subject["state"]
                if state == "complete":
                    actual_reason = subject["metadata"]["retire_reason"]
                    if not(actual_reason in ["blank","blank_consensus"]):
                        #retired as it should have been
                        pass
                    else:
                        #we retired something as non-blank when it actually blank - shouldn't happen
                        problem += 1
                        #print "problem with " + zooniverse_id
                        # print subject
                        # print overall_annotations[zooniverse_id].items()
                        # print actual_reason in ["blank","blank_consensus"]
                else:
                    unknown += 1
            else:
                completed_consensus_blank += 1
                #what was the actual reason the image was retired - if it was retired at all
                state = subject["state"]
                how_it_ended[zooniverse_id] = "consensus_blank"
                if state == "complete":
                    actual_reason = subject["metadata"]["retire_reason"]
                    #print actual_reason
                    #we retired an image because we thought it WAS blank
                    if not(actual_reason in ["blank","blank_consensus"]):
                        #error - false blank
                        false_completed_blank += 1
                    else:
                        #correctly retired a blank image
                        #completed_consensus_blank += 1
                        pass
                else:
                    pass
                    #print "="
                    unknown += 1

    elif sum(overall_annotations[zooniverse_id].values()) == 10:
        completed_set.add(zooniverse_id)
        subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
        state = subject["state"]
        how_it_ended[zooniverse_id] = "complete"
        regular_retire += 1
        completed += 1
        if state != "complete":
            #print "*"
            unknown += 1
    elif sum(overall_annotations[zooniverse_id].values()) == 4:
        continue
        #have these first 4 four people all said that the image does not contain a condor
        overall_contains_condor = False
        for ann in overall_annotations[zooniverse_id].keys():
            #does this set of annotations include a condor?
            contains_condor = False
            for animal,count in ann:
                contains_condor = contains_condor or ("condor" in animal)

            overall_contains_condor = overall_contains_condor or contains_condor

        #if none of the people reported a condor at all - retire it
        if not(overall_contains_condor):
            completed += 1
            completed_set.add(zooniverse_id)



print unknown
print false_blank
print false_completed_blank
print completed
print completed_blank
print completed_consensus_blank
print regular_retire


#pickle.dump(how_it_ended,open("/home/greg/condor_temp","w"))

# not_completed = [zooniverse_id for zooniverse_id in overall_annotations if not(zooniverse_id in completed_set)]
# for zooniverse_id in completed_set:
#     subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
#     if subject["state"] != "complete":
#         print subject
#         print overall_annotations[zooniverse_id]
#         pass