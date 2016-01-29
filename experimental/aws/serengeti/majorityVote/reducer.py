#!/usr/bin/env python
__author__ = 'greghines'

import sys

currPhoto = None
speciesList = ['','elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
currVotes = [0 for i in speciesList]

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    values = line.split("\t")
    photo = values[0]
    species = values[1][:-1]

    if photo != currPhoto:
        if currPhoto is not None:
            totalVotes = sum(currVotes)
            classification = [s for v,s in zip(currVotes,speciesList) if v >= (totalVotes/2.)]
            print photo + " " + str(classification)

        currVotes = [0 for i in speciesList]
        currPhoto = photo


    i = speciesList.index(species)
    currVotes[i] += 1

if currPhoto is not None:
    totalVotes = sum(currVotes)
    classification = [s for v,s in zip(currVotes,speciesList) if v >= (totalVotes/2.)]
    print photo + " " + str(classification)