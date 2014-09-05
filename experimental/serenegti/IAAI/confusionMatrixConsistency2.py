#!/usr/bin/env python
__author__ = 'greg'
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    baseDir = "/home/ggdhines/"
else:
    baseDir = "/home/greg/"

species = "wildebeest"
count = {}

with open(baseDir + "Downloads/expert_classifications_raw.csv","rU") as csvfile:
    goldreader = csv.reader(csvfile)
    next(goldreader, None)
    for line in goldreader:
        photoID = line[2]
        if not(photoID in count):
            count[photoID] = {}

        species = line[12]
        try:
            numAnimals = int(line[13])
        except ValueError:
            continue
        count[photoID][species] = numAnimals

wildebeestCount = {}

for photoID,classification in count.items():
    numAnimals = sum(classification.values())
    if classification.keys() == ["wildebeest"]:
        wildebeestCount[photoID] = numAnimals

correct = [0 for i in range(11)]
total = [0. for i in range(11)]

total = {}
correct = {}

values = []
userName = "Sobottka"
with open(baseDir +"Databases/goldMergedSerengeti.csv","rb") as csvfile:
    zooreader = csv.reader(csvfile,delimiter="\t")

    for l in zooreader:
        photoID,userID = l[0].split(",")
        if photoID in wildebeestCount:
            if not(userID in total):
                total[userID] = [0. for i in range(11)]
                correct[userID] = [0. for i in range(11)]

            classification = l[1].split(",")
            speciesList = [s.split(":")[0] for s in classification]

            if species in speciesList:
                correct[userID][wildebeestCount[photoID]] += 1

            if (userID == userName) and (wildebeestCount[photoID] == 10):
                if species in speciesList:
                    print "++ " + str(photoID)
                else:
                    print "-- " + str(photoID)

            total[userID][wildebeestCount[photoID]] += 1

userPercentage = [[] for i in range(0,11)]
for userID in total:
    if total[userID][1] == correct[userID][1]:
        continue
    if min(total[userID][1:]) == 0:
        continue
    if userID != userName:
        continue

    values.append((userID,correct[userID][1]/total[userID][1]))

    for n in range(1,11):
        if total[userID][n] == 0:
            continue

        userPercentage[n].append(correct[userID][n]/total[userID][n])

print values
print userPercentage
#for userID in userPercentage:
#    print userID,userPercentage[userID][1]

percentage = [np.median(p) for p in userPercentage[1:]]
std = [np.std(p) for p in userPercentage[1:]]
print 1-percentage[0]
print userPercentage
print correct[userName]
print total[userName]
#percentage = [c/t for (c,t) in zip(correct[1:],total[1:])]
plt.plot(range(1,11),percentage,'-o',color='black')
plt.plot(range(1,11), [1-(1-percentage[0])**n for n in range(1,11)],'--',color='black')
plt.xlabel("Number of Wildebeest in Picture")
plt.ylabel("Percentage of Correct Classifications")
plt.legend(('Actual','Expected'))
plt.show()

