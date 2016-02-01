__author__ = 'ggdhines'
import csv
from aggregation_api import AggregationAPI
import json

old_subjects = []

with open("/home/ggdhines/Dropbox/764_old/1224_PreProduction_Workflow/initDoes_this_look_like_a_pulsar.csv","rb") as old_subject_file:
    reader = csv.reader(old_subject_file)
    next(reader, None)

    for row in reader:
        old_subjects.append(int(row[0]))

with open("/home/ggdhines/Dropbox/764/1224_PreProduction_Workflow/initDoes_this_look_like_a_pulsar.csv","rb")as infile, open("/home/ggdhines/Downloads/dab0480f-1e81-4e2d-8403-e7cd81899a58 (1).csv","rb") as subject_file:
    reader = csv.reader(infile)
    next(reader, None)

    subjects = []
    for row in reader:
        if row[1] == "No":
            prob = 1-float(row[2])
        else:
            assert row[1] == "Yes"
            prob = float(row[2])
        subjects.append((row[0],prob,row[-1]))


    subjects.sort(key=lambda x:x[1],reverse=True)

    metadata = {}
    reader = csv.reader(subject_file)
    next(reader, None)

    for row in reader:
        # values = json.loads(row)
        id = int(row[0])
        metadata[id] = json.loads(row[-4])

    less_than_10 = 0
    with open("/tmp/pulsars.csv","wb") as f:
        for ii,(a,prob,c) in enumerate(subjects):
            # if str(a) != "1347097":
            #     continue


            # s = project.__get_subject_metadata__(a)
            if int(a) not in metadata:
                continue

            if int(a) in old_subjects:
                continue

            m = metadata[int(a)]["CandidateFile"]
            if "#Class" in metadata[int(a)]:
                print "skipping"
                continue

            if int(c) < 5:
                less_than_10 += 1

            # print prob,d
            # if prob == 1. and (int(d) == 1):
            #     print a
            #     assert False
            # print str(a)+","+str(prob)+","+str(c)+"," + "https://www.zooniverse.org/projects/zooniverse/pulsar-hunters/talk/subjects/"+a + "," + str(m)
            f.write(str(a)+","+str(prob)+","+str(c)+"," + "https://www.zooniverse.org/projects/zooniverse/pulsar-hunters/talk/subjects/"+a + "," + str(m) + "\n")

    print less_than_10