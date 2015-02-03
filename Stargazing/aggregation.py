#!/usr/bin/env python
__author__ = 'greg'
import json
import numpy as np


def score(annotations):
    # calculate the score associated with a given classification according to the algorithm
    # in the paper Galaxy Zoo Supernovae
    # an example of the json format used is
    # [{u'task': u'centered_in_crosshairs', u'value': 1}, {u'task': u'subtracted', u'value': 1}, {u'task': u'circular', u'value': 1}, {u'task': u'centered_in_host', u'value': 0}]
    assert annotations[0]["task"] == "centered_in_crosshairs"
    if annotations[0]["value"] == 0:
        return -1

    # they should have answered yes
    assert annotations[1]["task"] == "subtracted"
    if annotations[1]["value"] == 0:
        return -1

    assert annotations[2]["task"] == "circular"
    if annotations[2]["value"] == 0:
        return -1

    assert annotations[3]["task"] == "centered_in_host"
    if annotations[3]["value"] == 0:
        return 3
    else:
        return 1

# read in files named in the format classification-i.json
scores = []
for i in range(2):
    with open("/Users/greg/Downloads/classification-"+str(i)+".json") as f:
        data = json.load(f)
        scores.append(score(data["annotations"]))

aggregation = dict()
aggregation["mean_score"] = np.mean(scores)
# print out the mean score to aggregation.json
json.dump(aggregation,open("/Users/greg/Downloads/aggregation.json","wb"))

