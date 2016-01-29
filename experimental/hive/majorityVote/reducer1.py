#!/usr/bin/env python

import sys

currSubject = None
currUser = None
currSpeciesList = []


def speciesString(l):
    l.sort()
    if len(l) == 0:
        return ""

    retval = l[0]
    for s in l[1:]:
        retval += "_" + s

    return retval

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    tupleID, species = line.strip().split("\t")
    subject_zooniverse_id,user_name = tupleID.split(",")

    species = species[1:-1]
    user_name = user_name[1:-1]
    subject_zooniverse_id = subject_zooniverse_id[1:-1]

    if (subject_zooniverse_id != currSubject) or (user_name != currUser):
        if currSubject is not None:
            assert(currUser is not None)

            print currSubject + "\t" + currUser + "\t" + speciesString(currSpeciesList)

        currSubject = subject_zooniverse_id
        currUser = user_name
        currSpeciesString = ""
        currSpeciesList = []

    if not(species in currSpeciesList) and (species != ""):
        currSpeciesList.append(species)


if currSubject is not None:
    assert(currUser is not None)

    print currSubject + "\t" + currUser + "\t" + speciesString(currSpeciesList)