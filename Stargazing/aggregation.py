#!/usr/bin/env python
__author__ = 'greg'
import json
import numpy as np
import os
import sys
import math

# I (Greg) has slightly different directory structures depending on whether I am not the office or at home
# find out which computer I am on, and set the base directory
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

# import the Panoptes scripts which we use to download the raw data and upload the aggregated data
sys.path.append(base_directory+"/github/PanoptesScripts")
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
project_id = panoptesPythonAPI.get_project_id("Supernovae",token)
workflow_version = panoptesPythonAPI.get_workflow_id(project_id,token)
#assert False
#panoptesPythonAPI.find_aggregation(1,1,token)
#print panoptesPythonAPI.get_login_user_info(token)
#assert False
# dictionary to store the raw count for each score
scores = {}

# go through all of the classifications/annotations (terminology isn't completely consistent but not sure of what
# would be better)
for classification in panoptesPythonAPI.get_all_classifications(project_id,token):
    workflow_id = classification["links"]["workflow"]
    #print classification
    # if this is the first time we have encountered this subject, add it to the dictionary
    subject_id = classification["links"]["subjects"][0]
    if not((workflow_id,subject_id) in scores):
        scores[(workflow_id,subject_id)] = [0,0,0]

    # get the score index and increment that "box"
    annotations = classification["annotations"]
    scores[(workflow_id,subject_id)][score_index(annotations)] += 1


for (workflow_id,subject_id),values in scores.items():
    avg_score = (values[0]*-1+ + values[1]*1 + values[2]*3)/float(sum(values))
    std = math.sqrt((-1-avg_score)**2*(values[0]/float(sum(values))) + (1-avg_score)**2*(values[1]/float(sum(values))) + (3-avg_score)**2*(values[1]/float(sum(values))))
    aggregation = {"mean":avg_score,"std":std,"count":values}

    status,explanation = panoptesPythonAPI.create_aggregation(workflow_id,subject_id,token,aggregation)
    #print explanation
    if status == 400:
        print "updating"
        aggregation_id,etag = panoptesPythonAPI.find_aggregation_etag(workflow_id,subject_id,token)
        print etag
        panoptesPythonAPI.update_aggregation(workflow_id,workflow_version,subject_id,aggregation_id,token,aggregation,etag)




#print scores


