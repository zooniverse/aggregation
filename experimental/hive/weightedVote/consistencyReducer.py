#!/usr/bin/env python
import sys

current_zooniverse_id = None
consistencies = None

# input comes from STDIN (standard input)
for line in sys.stdin:
    subject_zooniverse_id, user_name, classification, voteDistributon = line.strip().split("\t")

    if (classification[0] == "\"") and (classification[-1] == "\""):
        classification = classification[1:-1]

    if current_zooniverse_id != subject_zooniverse_id:
        if current_zooniverse_id is not None:
            print current_zooniverse_id + "'\t" + consistencies

        current_zooniverse_id = subject_zooniverse_id
        consistencies = ""

    #convert voteDistribution
    consistency = 0
    numCategories = 0.

    for v in voteDistributon.split(" "):
        numCategories += 1
        found = False

        try:
            v = v.strip()
            c,f = v.split(":")
        except ValueError:
            sys.stderr.write(line+"\n")
            sys.stderr.write(voteDistributon+"\n")
            sys.stderr.write(v+"\n")
            sys.stderr.write(str(v.split(":"))+"\n")
            raise

        f = float(f)
        if c == classification:
            consistency += f
            found = True
        else:
            consistency += (1-f)

        if not found:
            sys.stderr.write(line+"\n")
            sys.stderr.write(classification+"\n")
            sys.stderr.write(voteDistributon+"\n")
            sys.stderr.write(c + "\n")
            assert found
    consistency = consistency/numCategories

    print subject_zooniverse_id + "\t" + user_name + "\t%.3f" % consistency