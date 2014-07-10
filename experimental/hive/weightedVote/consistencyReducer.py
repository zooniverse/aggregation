#!/usr/bin/env python
import sys

current_zooniverse_id = None
consistencies = None

# input comes from STDIN (standard input)
for line in sys.stdin:
    subject_zooniverse_id, user_name, classification, voteDistributon = line.split("\t")

    if current_zooniverse_id != subject_zooniverse_id:
        if current_zooniverse_id is not None:
            print current_zooniverse_id + "'\t" + consistencies

        current_zooniverse_id = subject_zooniverse_id
        consistencies = ""

    #convert voteDistribution
    consistency = 0
    numCategories = 0.
    found = False
    for v in voteDistributon.split(" "):
        numCategories += 1

        c,f = v.split(":")
        f = float(f)
        if c == classification:
            consistency += f
            found = True
        else:
            consistency += (1-f)

    assert found
    consistency = consistency/numCategories

    print subject_zooniverse_id + "\t" + user_name + "\t%.3f" % consistency