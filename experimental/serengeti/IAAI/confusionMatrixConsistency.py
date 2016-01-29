#!/usr/bin/env python
__author__ = 'greg'
import csv
import os
import matplotlib.pyplot as plt

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
        classification = line[12]
        if classification == species:
            try:
                count[photoID] = int(line[13])
            except ValueError:
                pass

correct = [0 for i in range(11)]
total = [0. for i in range(11)]

with open(baseDir +"Databases/goldMergedSerengeti.csv","rb") as csvfile:
    zooreader = csv.reader(csvfile,delimiter="\t")


    for l in zooreader:
        photoID,userID = l[0].split(",")
        if photoID in count:
            classification = l[1].split(",")
            speciesList = [s.split(":")[0] for s in classification]

            if species in speciesList:
                correct[count[photoID]] += 1

            total[count[photoID]] += 1

percentage = [c/t for (c,t) in zip(correct[1:],total[1:])]
plt.plot(range(1,11),percentage,color='black')
plt.plot(range(1,11), [1-0.223**n for n in range(1,11)],'--',color='black')
plt.xlabel("Number of Wildebeest in Picture")
plt.ylabel("Percentage of User Classifying a Photo as Containing a Wildebeest")
plt.legend(('Actual','Expected'))
plt.show()
