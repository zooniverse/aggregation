#!/usr/bin/env python
__author__ = 'greg'
import json
import numpy as np
import os
import sys

# I (Greg) has slightly different directory structures depending on whether I am not the office or at home
# find out which computer I am on, and set the base directory
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

# import the Panoptes scripts which we use to download the raw data and upload the aggregated data
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/github/PanoptesScripts")
import panoptesPythonAPI


# calculate the score associated with a given classification according to the algorithm
# in the paper Galaxy Zoo Supernovae
# an example of the json format used is
# [{u'task': u'centered_in_crosshairs', u'value': 1}, {u'task': u'subtracted', u'value': 1}, {u'task': u'circular', u'value': 1}, {u'task': u'centered_in_host', u'value': 0}]
def score_index(annotations):
    assert annotations[0]["task"] == "centered_in_crosshairs"
    if annotations[0]["value"] == 0:
        return 0  #-1

    # they should have answered yes
    assert annotations[1]["task"] == "subtracted"
    if annotations[1]["value"] == 0:
        return 0  #-1

    assert annotations[2]["task"] == "circular"
    if annotations[2]["value"] == 0:
        return 0  #-1

    assert annotations[3]["task"] == "centered_in_host"
    if annotations[3]["value"] == 0:
        return 2  #3
    else:
        return 1  #1

# get my userID and password
panoptes_file = open(base_directory+"/Databases/panoptes_login","rb")
userid = panoptes_file.readline()[:-1]
password = panoptes_file.readline()[:-1]

# get the token necessary to connect with panoptes
token = panoptesPythonAPI.get_bearer_token(userid,password)

# dictionary to store the raw count for each score
scores = {}

# go through all of the classifications/annotations (terminology isn't completely consistent but not sure of what
# would be better)
for classification in panoptesPythonAPI.get_all_classifications(2,token):
    # if this is the first time we have encountered this subject, add it to the dictionary
    subject_id = classification["links"]["subjects"][0]
    if not(subject_id in scores):
        scores[subject_id] = [0,0,0]

    # get the score index and increment that "box"
    annotations = classification["annotations"]
    scores[subject_id][score_index(annotations)] += 1

print scores


