#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    if i == 0:
        continue


    line = line.strip()
    words = line.split(",")

    subject_zooniverse_id = words[2][1:-1]
    user_name = words[1][1:-1]
    species = words[11][1:-1]
    if species == "":
        species = "None"
    capture_event_id = words[3][1:-1]

    if capture_event_id == "tutorial":
        continue

    print subject_zooniverse_id + "\t" +user_name + "\t" + species

