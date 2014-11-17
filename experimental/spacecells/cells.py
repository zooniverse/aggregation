#!/usr/bin/env python
import csv

with open("/home/greg/Databases/classifications.csv","r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for line in reader:
        words = line[3].split(",")
        index= words[0].rfind(":")
        print float(words[0][index+1:])
        break
