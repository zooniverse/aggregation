#!/usr/bin/env python

import sys

def speciesString(l):
    l.sort()
    if len(l) == 0:
        return ""

    retval = l[0]
    for s in l[1:]:
        retval += " " + s

    return retval

current_zooniverse_id = None
species_list = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
currentVotes = dict((s,0) for s in species_list)
totalVotes = 0

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    try:
        subject_zooniverse_id, classification = line.split("\t")
    except ValueError:
        subject_zooniverse_id, throwaway, classification = line.split("\t")

    attributes = classification[:-1].split(" ")

    if subject_zooniverse_id != current_zooniverse_id:
        if current_zooniverse_id != None:
            species_found = [s for s in species_list if currentVotes[s]>= totalVotes/2.]

            print current_zooniverse_id + "\t" + speciesString(species_found)

        current_zooniverse_id = subject_zooniverse_id
        currentVotes = dict((s,0) for s in species_list)
        totalVotes = 0

    for att in attributes:
        if att == "":
            continue

        currentVotes[att] += 1
    totalVotes += 1


species_found = [s for s in species_list if currentVotes[s]>= totalVotes/2.]

print current_zooniverse_id + "\t" + speciesString(species_found)
