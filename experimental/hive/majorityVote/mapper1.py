#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    try:
        (subject_zooniverse_id,user_name,species,capture_event_id) = line.strip().split()
    except ValueError:
        #assume that this

    if (capture_event_id == "\"tutorial\"") or (capture_event_id == "tutorial"):
        continue

    if species == "":
        species = "None"
    elif species == "\"\"":
        species = "\"None\""

    print subject_zooniverse_id + "," + user_name + "\t" + species
