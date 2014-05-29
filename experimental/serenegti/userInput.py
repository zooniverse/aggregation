#!/Users/greghines/Library/Enthought/Canopy_64bit/User/bin/python
__author__ = 'greghines'
import cPickle as pickle
import numpy
animalsList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']

expertDict = pickle.load(open('/Users/greghines/Databases/expertInput','rb'))

photoDict = pickle.load(open('/Users/greghines/Databases/userInput','rb'))

matchingDict = {}
overallMatching = [0. for i in range(25)]
for photoID in photoDict:
    #print "//// " + str(photoID)
    userCount = {animal:0 for animal in animalsList}
    numUsers = 0
    expertClassification = expertDict[photoID]
    matching = None
    numSpecies = []

    for classification in photoDict[photoID]:

        numUsers += 1
        if numUsers > 25:
            break
        #if len(classification) > 1:
        #    assert(('' in classification) == False)
        t = 0
        for a in classification:
            if a != '':
                userCount[a] += 1
                t += 1
        numSpecies.append(t)
        n = numpy.median(numSpecies)
        if n >= 1:
            currentClassification = sorted(userCount.keys(),key=lambda x:userCount[x],reverse=True)[0:int(n)]
        else:
            currentClassification = []




        #which ones do we detect?
        #currentClassification = []
        #for a in classification:
        #    if a != '':
        #        userCount[a] += 1
        #
        #for a in animalsList:
        #    if userCount[a] >= (numUsers/3.):
        #        currentClassification.append(a)
        matching = (sorted(currentClassification) == sorted(expertClassification))

        #    print sorted(userCount.keys(),key=lambda x:userCount[x],reverse=True)[:5]
        #print (sorted(currentClassification) , sorted(expertClassification))
        #print (currentClassification,sorted(expertClassification))
        if matching:
            overallMatching[numUsers-1] += 1.

    if matching == False:
        print photoID

    if matching:
        for n in range(numUsers,25):
            overallMatching[n]+=1


numPhotos = len(photoDict.keys())
overallMatching = [m/float(numPhotos) for m in overallMatching]

import matplotlib.pyplot as plt
plt.plot(range(25),overallMatching)
plt.show()
