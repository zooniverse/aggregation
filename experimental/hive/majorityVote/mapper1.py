#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    try:
        (subject_zooniverse_id,user_name,species,capture_event_id) = line.strip().split('\t')
    except ValueError:
        #assume that this
        sys.stderr.write(line)
        sys.stderr.write((str(line.strip().split('\t'))))
        raise

    if (user_name == "user_name") or (user_name == "\"user_name\""):
        continue

    if (capture_event_id == "\"tutorial\"") or (capture_event_id == "tutorial"):
        continue

    if ":" in species:
        #badly formed data
        continue

    if species == "":
        species = "None"
    elif species == "\"\"":
        species = "\"None\""

    print subject_zooniverse_id + "," + user_name + "\t" + species
