__author__ = 'ggdhines'
import csv

with open("/tmp/764/1099_Beta_Test_Workflow/initDoes_this_look_like_a_pulsar.csv","rb")as infile:
    reader = csv.reader(infile)
    next(reader, None)

    subjects = []
    for row in reader:
        subjects.append((row[0],row[2]))

    subjects.sort(key=lambda x:x[1])

    for a,b in subjects:
        print a,b