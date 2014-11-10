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
import datetime
import urllib


client = pymongo.MongoClient()
db = client['condor_2014-11-06']
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
num_users_before_retirement = []
#how_it_ended = pickle.load(open("/home/greg/condor_temp","r"))
strange = 0
cblank_to_blank = 0

blank_images = []
consensus_blank_images = []

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

#previously_completed = pickle.load(open("/home/greg/condor_temp","r"))

for i, classification in enumerate(condor_annotations.find()):
    if classification["subjects"] == []:
        continue

    if classification["created_at"] >= datetime.datetime(2014,9,15):
        continue


    zooniverse_id = classification["subjects"][0]["zooniverse_id"]

    if "tutorial" in condor_subjects.find_one({"zooniverse_id":zooniverse_id}).keys():
        continue

    if zooniverse_id in completed_set:
        continue

    keys = [ann.keys() for ann in classification["annotations"]]
    if ["marks"] in keys:
        ii = keys.index(["marks"])

        animal_count = {}
        for annotation_dict in classification["annotations"][ii]["marks"].values():
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

        user_annotations = tuple(sorted(animal_count.items(),key= lambda x:x[0]))
        #in case, all of the annotations have missed the animal key
        if user_annotations != ():
            if not(zooniverse_id in overall_annotations):
                overall_annotations[zooniverse_id] = {user_annotations:1}
            elif not(user_annotations in overall_annotations[zooniverse_id]):
                overall_annotations[zooniverse_id][user_annotations] = 1
            else:
                overall_annotations[zooniverse_id][user_annotations] += 1
    else:
        if not(zooniverse_id in overall_annotations):
            overall_annotations[zooniverse_id] = {(("blank",1),):1}
        elif not((("blank",1),) in overall_annotations[zooniverse_id]):
            overall_annotations[zooniverse_id][(("blank",1),)] = 1
        else:
            overall_annotations[zooniverse_id][(("blank",1),)] += 1









    #this should only happen in the race instance when nothing has been added for that image
    if not(zooniverse_id in overall_annotations):
        continue

    if overall_annotations[zooniverse_id] == {(("blank",1),):2}:
        #is this a false blank?
        completed_set.add(zooniverse_id)
        num_users_before_retirement.append(sum(overall_annotations[zooniverse_id].values()))
        # how_it_ended[zooniverse_id] = "blank"

        blank_images.append(zooniverse_id)

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

            #if how_it_ended[zooniverse_id] == "blank_consensus":
            #    cblank_to_blank += 1


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
            num_users_before_retirement.append(sum(overall_annotations[zooniverse_id].values()))
            subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})

            if non_blank_reason != []:
                #we retired it in the simulation because we thought it was non-blank
                regular_retire += 1
                #how_it_ended[zooniverse_id] = "consensus"
                #what was the actual reason the image was retired - if it was retired at all
                state = subject["state"]
                if state == "complete":
                    actual_reason = subject["metadata"]["retire_reason"]
                    if not(actual_reason in ["blank_consensus","no_condors_present"]):
                        #retired as it should have been
                        pass
                    else:
                        #we retired something as non-blank when it actually blank - shouldn't happen
                        strange += 1
                        #print "problem with " + zooniverse_id
                        # print subject
                        # print overall_annotations[zooniverse_id].items()
                        # print actual_reason in ["blank","blank_consensus"]
                else:
                    unknown += 1
            else:
                completed_consensus_blank += 1
                consensus_blank_images.append(zooniverse_id)
                #what was the actual reason the image was retired - if it was retired at all
                state = subject["state"]
                #how_it_ended[zooniverse_id] = "consensus_blank"
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
        num_users_before_retirement.append(sum(overall_annotations[zooniverse_id].values()))
        completed_set.add(zooniverse_id)
        subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
        state = subject["state"]
        #how_it_ended[zooniverse_id] = "complete"
        regular_retire += 1
        completed += 1
        reason = subject["metadata"]["retire_reason"]
        if not(reason in ["complete","consensus","no_condors_present"]):
            strange += 1
        if state != "complete":
            #print "*"
            unknown += 1
    elif sum(overall_annotations[zooniverse_id].values()) == 4:
        continue
        #have these first 4 four people all said that the image does not contain a condor
        has_anyone_found_a_condor = False
        for ann in overall_annotations[zooniverse_id].keys():
            #does this set of annotations include a condor?
            contains_condor = False
            for animal,count in ann:
                contains_condor = contains_condor or ("condor" in animal)

            has_anyone_found_a_condor = has_anyone_found_a_condor or contains_condor

        #if none of the people reported a condor at all - retire it
        if not(has_anyone_found_a_condor):
            completed += 1
            completed_set.add(zooniverse_id)



print unknown
print false_blank
print false_completed_blank
print completed
print completed_blank
print completed_consensus_blank
print regular_retire
print np.mean(num_users_before_retirement)
print np.median(num_users_before_retirement)
print strange
#pickle.dump(how_it_ended,open("/home/greg/condor_temp","w"))

# not_completed = [zooniverse_id for zooniverse_id in overall_annotations if not(zooniverse_id in completed_set)]
# #for zooniverse_id in completed_set:
# gah = 0
# for subject in condor_subjects.find({"state":"complete"}):
#     zooniverse_id = subject["zooniverse_id"]
#     if not(zooniverse_id in completed_set):
#         gah += 1
#         #print overall_annotations[zooniverse_id]
#         # try:
#         #     print subject["metadata"]["retire_reason"]
#         # except KeyError:
#         #     gah += -1
#
#
# print gah
#
# to_sample = random.sample(blank_images,100)
# for zooniverse_id in to_sample:
#     subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
#     location = subject["location"]["standard"]
#     slash_index = location.rfind("/")
#     f_name = location[slash_index+1:]
#     if not(os.path.isfile(base_directory+"/Databases/condors/images/blank/initial/"+f_name)):
#             urllib.urlretrieve ("http://www.condorwatch.org/subjects/standard/"+f_name, base_directory+"/Databases/condors/images/blank/initial/"+f_name)
#
# to_sample = random.sample(consensus_blank_images,100)
# for zooniverse_id in to_sample:
#     subject = condor_subjects.find_one({"zooniverse_id":zooniverse_id})
#     location = subject["location"]["standard"]
#     slash_index = location.rfind("/")
#     f_name = location[slash_index+1:]
#     if not(os.path.isfile(base_directory+"/Databases/condors/images/blank/consensus/"+f_name)):
#             urllib.urlretrieve ("http://www.condorwatch.org/subjects/standard/"+f_name, base_directory+"/Databases/condors/images/blank/consensus/"+f_name)