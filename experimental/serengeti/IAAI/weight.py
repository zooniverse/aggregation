#!/usr/bin/env python
import csv
#ASG000pt52,merxator     wildebeest

photos = {}
beta = 1
def weight(TP,TN,FP,FN):
    if (TP+beta*TN + FP+FN) == 0:
        return -1
    return (TP+beta*TN)/float(TP+beta*TN + FP+FN)

searchFor = "zebra"

with open("/home/greg/Databases/goldMergedSerengeti.csv") as f:
    reader = csv.reader(f,delimiter="\t")

    for meta,speciesList in reader:
        photoID,userID = meta.split(",")
        animals = [s.split(":")[0] for s in speciesList.split(",")]

        if userID == "pjeversman":
            if searchFor in animals:
                photos[photoID] = True
            else:
                photos[photoID] = False

TP = 0.
TN = 0.
FP = 0.
FN = 0.

weightValues = []

with open("/home/greg/Downloads/Expert_Classifications_For_4149_S4_Captures.csv") as f:
    reader = csv.reader(f)
    next(reader, None)

    for photoID,image,count,s1,s2,s3 in reader:
        if photoID in photos:
            if (searchFor in [s1,s2,s3]):
                if photos[photoID]:
                    TP += 1
                else:
                    FN += 1
            else:
                if photos[photoID]:
                    FP += 1
                else:
                    TN += 1

            weightValues.append(weight(TP,TN,FP,FN))

print TP,TN,FP,FN
print photos

import matplotlib.pyplot as plt
plt.plot(range(len(weightValues)),weightValues)
plt.ylim(0.5,1.1)
plt.xlabel(str(beta))
plt.show()