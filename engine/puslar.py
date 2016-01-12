__author__ = 'ggdhines'
import csv
from aggregation_api import AggregationAPI

with open("/tmp/764/1224_PreProduction_Workflow/initDoes_this_look_like_a_pulsar.csv","rb")as infile, AggregationAPI(764,"development") as project:
    project.__setup__()
    reader = csv.reader(infile)
    next(reader, None)

    subjects = []
    for row in reader:
        subjects.append((row[0],row[2],row[-1]))

    subjects.sort(key=lambda x:x[1])

    for a,b,c in subjects:
        s = project.__get_subject_metadata__(a)
        m = s["subjects"][0]["metadata"]["CandidateFile"]

        print a,b,c,"https://www.zooniverse.org/projects/zooniverse/pulsar-hunters/talk/subjects/"+a,m