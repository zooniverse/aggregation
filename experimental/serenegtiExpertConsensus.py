#!/Users/greghines/Library/Enthought/Canopy_64bit/User/bin/python
__author__ = 'greghines'
import csv
import cPickle as pickle

goldStandardDict = {}
resultingDict = {}

print "loading experts"
i = 0
with open('/Users/greghines/Databases/expert_classifications.csv', 'rb') as csvfile:
    classificationReader = csv.reader(csvfile,delimiter=',')
    next(classificationReader,None)

    for row in classificationReader:
        photoID = row[2]
        species = row[12]
        expertID = row[1]
        i+=1
        if photoID in goldStandardDict:
            if (expertID in goldStandardDict[photoID]):
                goldStandardDict[photoID][expertID].append(species)
            else:
                goldStandardDict[photoID][expertID] = [species,]
        else:
            goldStandardDict[photoID] = {expertID:[species,]}


for photoID in goldStandardDict:
    animalDict = {}
    numExperts = 0
    expertSpecies = []
    for expertId in goldStandardDict[photoID]:
        numExperts += 1
        for animal in goldStandardDict[photoID][expertId]:
            if animal in animalDict:
                animalDict[animal] += 1
            else:
                animalDict[animal] = 1

    for animal in animalDict:
        if animalDict[animal] >= (numExperts /2.):
            expertSpecies.append(animal)

    resultingDict[photoID] = expertSpecies[:]

print "done"
pickle.dump(resultingDict,open('/Users/greghines/Databases/expertInput','wb'))


