#!/usr/bin/env python

import sys

current_zooniverse_id = None
votesDict = {}


def toStr(voteDistribution):
    s = ""
    t = float(len(voteDistribution))

    for classification in voteDistribution:
        if s == "":
            s = classification + ":%.3f" % (voteDistribution[classification]/t)
        else:
            s += " " + classification + ":%.3f" % (voteDistribution[classification]/t)

    return s

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    try:
        subject_zooniverse_id, classification, consistency = line.split("\t")
        consistency = float(consistency)
    except ValueError:
        subject_zooniverse_id, classification = line.split("\t")
        consistency = 1.

    classification = classification.strip()

    if subject_zooniverse_id != current_zooniverse_id:
        if current_zooniverse_id is not None:
            print current_zooniverse_id + "\t" + toStr(votesDict)

        votesDict = {}
        current_zooniverse_id = subject_zooniverse_id

    if not(classification in votesDict):
        votesDict[classification] = consistency
    else:
        votesDict[classification] += consistency


if current_zooniverse_id is not None:
    print current_zooniverse_id + "\t" + toStr(votesDict)