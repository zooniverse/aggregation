#!/usr/bin/env python
__author__ = 'greghines'
import csv
import time

goldStandardDict = {}
photoCount = {}
timeDict = {}

#try to open the file first on my laptop, then try on my desktop
try:
    csvfile = open('/Users/greghines/Databases/expert_classifications.csv', 'rb')
    classificationReader = csv.reader(csvfile, delimiter=',')
    next(classificationReader,None)

except IOError:
    csvfile = open('/home/ggdhines/Databases/serengeti/expert_classifications_raw.csv', 'rU')
    classificationReader = csv.reader(csvfile, delimiter=',')

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



#now go through the actual classifications
#again, go try to open on my laptop first
try:
    csvfile = open('/Users/greghines/Downloads/2014-05-18_serengeti_classifications.csv', 'rb')
except IOError:
    csvfile = open('/home/ggdhines/Databases/serengeti/2014-05-25_serengeti_classifications.csv','rb')

print csvfile.readline()[:-1]
#classificationReader = csv.reader(csvfile,delimiter=',')
while True:
    line = csvfile.readline()
    if not line: break
    row = line.split(",")
    photoID = row[2][1:-1]
    if photoID in goldStandardDict:
        print line[:-1]



    #for row in classificationReader:
    #    print row
    #    break
    #     photoID = row[2]
    #     user_name = row[1]
    #     dateStr = row[4]
    #     #'2012-12-10 23:37:51 UTC'
    #     d = time.strptime(dateStr,'%Y-%m-%d %H:%M:%S %Z')
    #     #if row[6] == "tutorial":
    #     #    continue
    #     seconds = calendar.timegm(d)
    #
    #     if (photoID in photoCount):
    #         if not(user_name in photoCount[photoID]):
    #             photoCount[photoID].append(user_name)
    #             timeDict[photoID].append(seconds)
