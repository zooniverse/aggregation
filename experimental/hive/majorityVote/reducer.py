#!/usr/bin/env python

import sys

current_zooniverse_id = None
species_list = ['None','elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
currentVotes = dict((s,0) for s in species_list)
totalVotes = 0

# input comes from STDIN (standard input)
for i,line in enumerate(sys.stdin):
    subject_zooniverse_id, user_name, species = line.strip().split("\t")
    species = species[:-1]

    if subject_zooniverse_id != current_zooniverse_id:
        if current_zooniverse_id != None:
            species_found = [s for s in species_list if currentVotes[s]>= totalVotes/2.]
            if species_found == []:
                species_found = [""]

            if (len(species_found) > 1) and (species_found[0] == ""):
                del species_found[0]

            if totalVotes >= 10:
                print current_zooniverse_id + "\t" + str(species_found)

        current_zooniverse_id = subject_zooniverse_id
        currentVotes = dict((s,0) for s in species_list)
        totalVotes = 0

    currentVotes[species] += 1
    totalVotes += 1


species_found = [s for s in species_list if currentVotes[s]>= totalVotes/2.]
if species_found == []:
    species_found = [""]

if totalVotes >= 10:
    print current_zooniverse_id + "\t" + str(species_found)