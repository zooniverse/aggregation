#!/usr/bin/env python

import sys

current_zooniverse_id = None
species_list = ['None','elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
attVotes = dict((s,0) for s in species_list)
totalVotes = 0


def fractionString(totalVotes,v):
    s = ""

    for att in v:
        if v[att] == 0:
            continue
        else:
            if s == "":
                s = att+":"+"%.2f"%(v[att]/float(totalVotes))
            else:
                s += " " + att+":"+"%.2f"%(v[att]/float(totalVotes))

    return s



# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    try:
        subject_zooniverse_id, classification = line.split("\t")
    except ValueError:
        subject_zooniverse_id, throwaway, classification = line.split("\t")

    attributes = classification[:-1].split(" ")

    if subject_zooniverse_id != current_zooniverse_id:
        if current_zooniverse_id != None:
            print current_zooniverse_id + "\t" + fractionString(totalVotes, attVotes)

        current_zooniverse_id = subject_zooniverse_id
        totalVotes = 0
        attVotes = dict((s,0) for s in species_list)

    for att in attributes:
        attVotes[att] += 1
    totalVotes += 1

print current_zooniverse_id + "\t" + fractionString(totalVotes, attVotes)
