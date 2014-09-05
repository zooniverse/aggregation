#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

numUser = [5,10,15,20,25]
algPercent = []
currPercent = []
speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']


correct = {"wildebeest":[],"zebra":[],"hartebeest":[],"gazelleThomsons":[],"buffalo":[],"impala":[],"warthog":[],"giraffe":[],"elephant":[],"human":[],"gazelleGrants":[],"guineaFowl":[],"hyenaSpotted":[],"otherBird":[],"hippopotamus":[],"reedbuck":[],"eland":[],"baboon":[],"lionFemale":[]}

for j in range(1):
    print j
    photos,users = setup(tau=50)

    for p in photos.values():
        p.__sample__(25)
    for u in users.values():
        u.__prune__()

    #initialize things using majority voting
    for p in photos.values():
        p.__majorityVote__()

    #estimate the user's "correctness"
    for u in users.values():
        for s in speciesList:
            u.__speciesCorrect__(s,beta=0.01)

    for p in photos.values():
        p.__currAlg__()


    for s in correct.keys():
        correctCount = 0
        for p in photos.values():
            if (s in p.goldStandard) and (s in p.contains):
                correctCount += 1

        correct[s].append(correctCount)

for s,c in correct.items():
    print s,np.mean(c)
