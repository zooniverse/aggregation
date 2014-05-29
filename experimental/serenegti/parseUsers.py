#!/Users/greghines/Library/Enthought/Canopy_64bit/User/bin/python
__author__ = 'greghines'
import csv
import cPickle as pickle

currentPhoto = None
currentUser = None
speciesList = []
f = []

photoDict = {}

with open('/Users/greghines/Databases/filteredSerenegti.csv', 'rb') as csvfile:
    classificationReader = csv.reader(csvfile,delimiter=',')
    next(classificationReader,None)

    for row in classificationReader:
        newPhoto = row[2]
        newUser = row[1]
        species = row[11]

        if (newPhoto != currentPhoto) or (newUser != currentUser):
            if currentPhoto != None:
                if currentPhoto in photoDict:
                    photoDict[currentPhoto].append(speciesList[:])
                else:
                    photoDict[currentPhoto] = [speciesList[:]]
                #f.append((currentPhoto[:],currentUser[:],speciesList[:]))

            currentPhoto = newPhoto
            currentUser = newUser
            speciesList = [species]
        else:
            speciesList.append(species)

if currentPhoto in photoDict:
    photoDict[currentPhoto].append(speciesList[:])
else:
    photoDict[currentPhoto] = [speciesList[:]]

pickle.dump(photoDict,open('/Users/greghines/Databases/userInput','wb'))
