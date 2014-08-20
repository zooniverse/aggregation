#!/usr/bin/env python
__author__ = 'greg'
import csv
from itertools import chain, combinations
import numpy as np
import os
import random

speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']
#speciesList = [['buffalo','wildebeest']]#,'zebra','elephant','warthog','impala','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']

ss = 0
def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def map2(classification,speciesFilter):
    for gIndex, group in enumerate(powerset(speciesFilter)):
        g_complement = [s for s in speciesFilter if not(s in group)]

        if not(set(classification).intersection(group) == set(group)):
            pass
        elif not(set(classification).intersection(g_complement) == set()):
            pass
        else:
            return gIndex

    assert(False)

class MoreThanOneDifference(Exception):
    def __init__(self):
        pass
    def __str__(self):
        pass

class Photo:
    def __init__(self,tau=None):
        self.users = []
        self.users2 = []
        self.mostlikely = None
        self.updated = False
        self.numGroups = None
        self.speciesFilter = None
        self.goldStandard = None
        self.likelihood = {species:1 for species in speciesList}
        self.useGoldStandard = False

        if tau is None:
            self.tau = 10#21
        else:
            self.tau = tau

    def __gowithMostLikely2__(self):
        if self.contains == []:
            return None
        t = 0
        e = 0
        for species in self.contains:
            overWeight = 0
            enoughVotes = [u for u in self.users if (len(u.classifications) >= 1) and (u.__nonempty__(self)) and (u.speciesCorrect[species] > 0)]
            weights = {u:u.speciesCorrect[species]**self.tau for u in enoughVotes}

            if weights == {}:
                mostLikely = self.users
            else:
                #allow for more then one user having the maximum weight
                mostLikely = [u for u,w in weights.items() if w == max(weights.values())]

            speciesInPhoto = [species in u.__getClassification__(self) for u in mostLikely]

            match = [(species in self.contains) == s for s in speciesInPhoto]

            if not(False in match):
                t += 1

        return t/float(len(self.contains))

    def __gowithMostLikely__(self):
        t = 0.
        c = 0
        for species in speciesList:
            overWeight = 0
            enoughVotes = [u for u in self.users if (len(u.classifications) >= 1) and (u.__nonempty__(self)) and (u.speciesCorrect[species] > 0)]
            weights = {u:u.speciesCorrect[species]**self.tau for u in enoughVotes}

            if weights == {}:
                mostLikely = self.users
            else:
                #allow for more then one user having the maximum weight
                mostLikely = [u for u,w in weights.items() if w == max(weights.values())]

            #contains = [species in u.classifications]

            speciesInPhoto = [species in u.__getClassification__(self) for u in mostLikely]
            if True in speciesInPhoto:
                t += 1
                if species in self.contains:
                    c += 1


        if t == 0:
            return None
        return c/t




    def __currAlg__(self):
        numSpecies = []
        votes = {s: 0 for s in speciesList}
        for u in self.users:
            c = u.classifications[self]
            if c == ["none"]:
                pass
                #numSpecies.append(0)
            else:
                numSpecies.append(len(c))
                for s in c:
                    votes[s] += 1

        if numSpecies == []:
            self.contains = []
        else:
            self.contains = sorted(speciesList, key = lambda x:votes[x], reverse = True)[:int(np.median(numSpecies))]

    def __useGoldStandard__(self):
        self.useGoldStandard = True

    def __adduser__(self,user):
        assert(type(user) != str)
        self.users.append(user)
        self.users2.append(user)

    def __sample__(self,limit):
        if limit < len(self.users):
            self.users = random.sample(self.users2,limit)
        else:
            self.users = self.users2[:]

    def __canAdd__(self):
        return len(self.users) < self.limit
        #return True



    def __addGoldStandard__(self,classification):
        if self.goldStandard is None:
            self.goldStandard = [classification]
        elif not(classification in self.goldStandard):
            self.goldStandard.append(classification)

    def __getPrediction__(self):
        return self.contains

    def __getPrediction2__(self,f=None):
        prediction= set([s for s in speciesList if self.contains[s]])
        if f is None:
            return prediction
        else:
            return set([s for s in prediction if s in f])

    def __getmostlikely__(self):
        assert(self.mostlikely is not None)
        return self.mostlikely

    def __wasupdated__(self):
        return self.updated

    def __setSpeciesFilter__(self,speciesFilter):
        self.numGroups = 2**(len(speciesFilter))
        self.speciesFilter = speciesFilter
        self.mostlikely = None

    def __goldStandardCompare__(self):
        # diff1 = set([s for s in self.contains if not(s in self.goldStandard)])
        # diff2 = set([s for s in self.goldStandard if not(s in self.contains)])
        # if set(self.goldStandard) != set(self.contains):
        #     print (self.contains,self.goldStandard)
        #     for u in self.users:
        #         print (u.__getClassification__(self),u.pCorrect)
        #         #print u.count[self]
        #     print "///"
        #     print self.contains
        #     print "missed " + str(diff2)
        #     print sorted(self.likelihood.items(), key=lambda x:x[1],reverse=True)[0:10]
        #     #for u in self.users:
        #     #    print u.__getClassification__(self)
        #
        #     m = self.__majorityVote__()
        #     print m[0:10]
        #         #print [species for species in speciesList if (tagCount[species] >= len(self.users)/2.)]
        #     print "==="
        return set(self.goldStandard) == set(self.contains)

    def __majorityVote__(self):
        if self.useGoldStandard:
            self.contains = self.goldStandard[:]

        tagCount = {s:0 for s in speciesList}
        for user in self.users:
            tags = set(user.__getClassification__(self))
            for s in tags:
                if s != 'none':
                    tagCount[s] += 1
        assert(max(tagCount.values()) <= len(self.users))

        nonEmptyCount = sum([1 for u in self.users if u.__nonempty__(self)])
        self.contains = [species for species in speciesList if (tagCount[species] >= nonEmptyCount/2.)]

        return sorted([(s,tagCount[s]/float(len(self.users))) for s in tagCount],key = lambda x:x[1],reverse=True)

    def __weightedMajorityVote2__(self):
        #print self.contains
        #print self.goldStandard

        enoughVotes = [u for u in self.users if (len(u.classifications) >= 30) and (u.__nonempty__(self))]
        weights = {u:u.pCorrect for u in enoughVotes}
        tagCount = {s:0 for s in speciesList}
        for user in enoughVotes:
            tags = set(user.__getClassification__(self))
            for s in tags:
                if s != 'none':
                    tagCount[s] += weights[user]
        #nonEmptyWeight = sum([u.pCorrect for u in enoughVotes if u.__nonempty__(self)])
        self.contains = [species for species in speciesList if (tagCount[species] >= sum(weights.values())/2.)]
        #print self.contains
        #print "===---"
        #numClassifications = [len(u.classifications) for u in self.users]
        #print [w if n >= 30 else 0.5 for w,n in zip(weights,numClassifications)]

    def __weightedMajorityVote__(self):
        if self.useGoldStandard:
            self.contains = self.goldStandard[:]

        #print self.contains
        #print self.goldStandard
        self.contains = []
        for species in speciesList:
            overWeight = 0
            enoughVotes = [u for u in self.users if (len(u.classifications) >= 1) and (u.__nonempty__(self)) and (u.speciesCorrect[species] > 0)]
            weights = {u:u.speciesCorrect[species]**self.tau for u in enoughVotes}

            if enoughVotes == []:
                enoughVotes = [u for u in self.users if (len(u.classifications) >= 1) and (u.__nonempty__(self))]
                weights = {u:1 for u in enoughVotes}

            if enoughVotes == []:
                continue
            #assert(enoughVotes != [])

            overallWeight = sum([w for u,w in weights.items() if species in u.__getClassification__(self)])
            if overallWeight >= sum(weights.values())/2.:
                self.contains.append(species)

        # if set(self.contains) != set(self.goldStandard):
        #     print self.contains
        #     print self.goldStandard
        #     enoughVotes = [u for u in self.users if (len(u.classifications) >= 30) and (u.__nonempty__(self)) and (u.speciesCorrect["ostrich"] != -1)]
        #     print enoughVotes
        #     weights =  {u:u.speciesCorrect["ostrich"] for u in enoughVotes}
        #     print weights
        #     overallWeight = sum([w for u,w in weights.items() if "ostrich" in u.__getClassification__(self)])
        #     print overallWeight
        #     print sum(weights.values())/2
        #     print "=-==="
        #     for u in self.users:
        #         print u.__getClassification__(self)
        #     assert False

        return




class User:
    def __init__(self):
        self.classifications = {}
        self.classifications2 = {}
        self.mappedClassifications = {}
        self.confusion = {}
        self.filter = None
        self.e = 0
        self.globalConfusions = None
        self.processed = []
        self.count = {}
        self.pCorrect = None
        self.speciesCorrect = {}

    def __prune__(self):
        self.classifications = {}
        for photo in self.classifications2.keys():
            if self in photo.users:
                self.classifications[photo] = self.classifications2[photo]

    def __speciesCorrect__(self,species):
        posTotal =  0.
        posCorrect = 0.
        negTotal = 0.
        negCorrect = 0.
        correct = 0.
        incorrect = 0.

        for p,c in self.classifications.items():
            if (species in c) and (species in p.goldStandard):
                correct += 1
            elif (species in c) or (species in p.goldStandard):
                incorrect += 1
            else:
                correct += 0.0

        if (correct + incorrect) == 0:
            self.speciesCorrect[species] = -1
            return -1
        else:
            self.speciesCorrect[species] = correct/(correct+incorrect)
            return correct/(correct+incorrect)

    def __getStats__(self):
        easyTotal = 0.
        easyCorrect = 0.
        hardTotal = 0.
        hardCorrect = 0.

        for p,c in self.classifications.items():
            if set(c) == set(p.goldStandard):
                if set(p.goldStandard) == set(p.contains):
                    easyCorrect += 1
                    easyTotal += 1
                else:
                    hardCorrect += 1
                    hardTotal += 1
            else:
                if set(p.goldStandard) == set(p.contains):
                    easyTotal += 1
                else:
                    hardTotal += 1

        if (hardTotal+easyTotal) <= 3:
            #print "only one"
            return -1,-1

        if hardTotal <= 4:
            #print "no hard"
            return easyCorrect/easyTotal, -1
        elif easyTotal == 0:
            #print "no easy"
            return -1,hardCorrect/hardTotal
        else:
            #print "easy and hard"
            return easyCorrect/easyTotal, hardCorrect/hardTotal

    def __correctnessProbability__(self):
        correct = 0
        for p,c in self.classifications.items():
            if set(c) == set(p.contains):
                correct += 1

        self.pCorrect = correct/float(len(self.classifications))


    def __globalConfusions__(self,g):
        self.globalConfusions = g

    def __nonempty__(self,photoNode):
        return self.classifications[photoNode] != ["none"]

    def __addClassification__(self,photoID,photoNode,classification):
        c2 = [cv.split(":")[0] for cv in classification]
        self.classifications[photoNode] = c2[:]
        self.classifications2[photoNode] = c2[:]
        #self.count[photoNode] = [cv.split(":") for cv in classification]
        #assert not(photoID in self.processed)
        #self.processed.append(photoID)

    def __getProbability__(self,photoNode,species):
        p = [self.confusion[species][species2] for species2 in self.classifications[p]]

    def __getProbability2__(self,photoNode,species):
        if self.confusion == {}:
            if species in self.classifications[photoNode]:
                return 1
        elif not(species in self.confusion):
            overlap = [s for s in self.globalConfusions[species] if s in self.classifications[photoNode]]
            if overlap == []:
                return 0.1
            else:
                return 0.2
        else:
            #what animals is the user likely to confuse with species?
            #which of those animals do they user report?
            overlap = [s for s in self.confusion[species] if s in self.classifications[photoNode]]
            if overlap == []:
                overlap = [s for s in self.globalConfusions[species] if s in self.classifications[photoNode]]
                if overlap == []:
                    return 0.1
                else:
                    return 0.2

            else:
                return sum([self.confusion[species][s] for s in overlap])

            #return maxLikelihood

    def __getClassification__(self,photoID):
        return self.classifications[photoID]



    def __updateConfusion__(self):
        confusions = {}
        self.confusion = {}
        self.reverseConfusion = {}
        e = 0
        for photoNode,classification in self.classifications.items():
            if classification == ['none']:
                continue

            prediction = photoNode.__getPrediction__()
            diff1 = [s for s in classification if not(s in prediction)]
            diff2 = [s for s in prediction if not(s in classification)]

            if (len(diff1) == 1) and (len(diff2) == 1):
                d1 = diff1[0]
                d2 = diff2[0]


                #if (len(classification) != 1) or (len(prediction) != 1):
                #    print (d1,d2)
                if not(d2 in confusions):
                    confusions[d2] = set([d1])
                elif not(d1 in confusions[d2]):
                    confusions[d2].add(d1)

                if not(d2 in self.confusion):
                    self.confusion[d2] = {d1:1}
                    self.reverseConfusion[d1] = {d2:1}
                elif not(d1 in self.confusion[d2]):
                    self.confusion[d2][d1] = 1
                    self.reverseConfusionconfusion[d1][d2] = 1
                else:
                    self.reverseConfusionconfusion[d1][d2] += 1

            elif (len(classification) == 1) and (len(diff2) == 1):
                if (classification[0] != "otherBird") and (diff2[0] != "otherBird"):
                    d1 = classification[0]
                    d2 = diff2[0]

                    if not(d2 in self.confusion):
                        self.confusion[d2] = {d1:1}
                    elif not(d1 in self.confusion[d2]):
                        self.confusion[d2][d1] = 1
                    else:
                        self.confusion[d2][d1] += 1


            correct = [s for s in classification if s in prediction]
            for c in correct:
                if not(c in self.confusion):
                    self.confusion[c] = {c:1}
                    self.reverseConfusion[c] = {c:1}
                elif not(c in self.confusion[c]):
                    self.confusion[c][c] = 1
                    self.reverseConfusion[c][c] = 1
                else:
                    self.confusion[c][c] += 1
                    self.reverseConfusiononfusion[c][c] += 1


        #now normalize
        for a in self.confusion:
            summed = float(sum(self.confusion[a].values()) )
            for c in self.confusion[a]:
                self.confusion[a][c] /= summed
            #assert(sum(self.confusion[a].values()) == 1)

        #print self.confusion
        return confusions

    def __getConfusion__(self):
        return self.confusion

    def __getErrors__(self):
        errors = []
        for photoNode,reported in self.classifications.items():
            actual = photoNode.__getPrediction__()
            diff1 = [s for s in reported if not(s in actual)]
            diff2 = [s for s in actual if not(s in reported)]

            if diff1 == ['none']:
                continue

            if (diff1 != []) or (diff2 != []):
                if (len(diff1) != 1) or (len(diff2) != 1):
                    raise MoreThanOneDifference()
                else:
                    errors.append((diff2[0],diff1[0]))

        return errors


def setup(limit=None,tau=None):
    photos = {}
    users = {}

    processed = []

    if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
        baseDir = "/home/ggdhines/"
    else:
        baseDir = "/home/greg/"

    with open(baseDir +"Databases/goldMergedSerengeti.csv","rb") as csvfile:
        zooreader = csv.reader(csvfile,delimiter="\t")


        for l in zooreader:
            photoID,userID = l[0].split(",")
            classification = l[1].split(",")

            if not(photoID in photos):
                photos[photoID] = Photo(tau=tau)
            #elif not(photos[photoID].__canAdd__()):
            #    continue

            if not(userID in users):
                users[userID] = User()


            photos[photoID].__adduser__(users[userID])
            users[userID].__addClassification__(photoID,photos[photoID],classification)





    with open(baseDir + "Downloads/expert_classifications_raw.csv","rU") as csvfile:
        goldreader = csv.reader(csvfile)
        next(goldreader, None)
        for line in goldreader:
            photoID = line[2]
            classification = line[12]
            photos[photoID].__addGoldStandard__(classification)

    return photos,users