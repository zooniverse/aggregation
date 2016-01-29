#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    (subject_zooniverse_id,user_name,species,capture_event_id) = line.strip().split()

    if species == "":
        species = "None"

    if capture_event_id == "\"tutorial\"":
        continue

    print subject_zooniverse_id[1:-1] + "\t" + user_name[1:-1] + "\t" + species[1:-1]
