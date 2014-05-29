#!/Users/greghines/Library/Enthought/Canopy_64bit/User/bin/python
__author__ = 'greghines'
import csv
import time

goldStandardDict = {}
photoCount = {}
timeDict = {}

print "loading experts"
with open('/Users/greghines/Databases/expert_classifications.csv', 'rb') as csvfile:
    classificationReader = csv.reader(csvfile,delimiter=',')
    next(classificationReader,None)

    for row in classificationReader:
        photoID = row[2]
        species = row[12]

        if photoID in goldStandardDict:
            if not (species in goldStandardDict[photoID]):
                goldStandardDict[photoID].append(species)
        else:
            goldStandardDict[photoID] = [species]
            photoCount[photoID] = []
            timeDict[photoID] = []

print "loading regular users"
import calendar
#now go through the actual classifications
with open('/Users/greghines/Databases/filteredSerenegti.csv', 'rb') as csvfile:
    classificationReader = csv.reader(csvfile,delimiter=',')
    next(classificationReader,None)

    for row in classificationReader:
        photoID = row[2]
        user_name = row[1]
        dateStr = row[4]
        #'2012-12-10 23:37:51 UTC'
        #d = time.strptime(dateStr,'%Y-%m-%d %H:%M:%S %Z')
        #if row[6] == "tutorial":
        #    continue
        #if (d.tm_year < 2013) or ((d.tm_year == 2013) and (d.tm_mon < 4)):
        #    continue
        #seconds = calendar.timegm(d)

        if (photoID in photoCount):
            if not(user_name in photoCount[photoID]):
                photoCount[photoID].append(user_name)



l = [len(photoCount[photoID]) for photoID in photoCount]
print "printing graph"
# timeDiff = []
# for photoID in timeDict:
#     history = timeDict[photoID]
#     if len(history) > 25:
#         assert((history[-1]-history[24]) > 0)
#         timeDiff.append((history[-1]-history[24])/86400.)
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(l, 10, normed=1, facecolor='green', alpha=0.5)
plt.show()
