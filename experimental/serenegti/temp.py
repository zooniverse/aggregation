#!/Users/greghines/Library/Enthought/Canopy_64bit/User/bin/python
import csv
__author__ = 'greghines'

with open('/Users/greghines/Downloads/2014-05-18_serengeti_classifications.csv', 'rb') as csvfile:
    classificationReader = csv.reader(csvfile,delimiter=',')
    next(classificationReader,None)

    for row in classificationReader:
        if row[2] == "ASG000blh8":
            print row[11]