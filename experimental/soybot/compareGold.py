#!/usr/bin/env python
__author__ = 'greg'
import pymongo



client = pymongo.MongoClient()
db = client['penguin_2015-01-18']
collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

subjects = subject_collection.find({"metadata.path":{"$regex":"MAIVb2012a"}})

with open("/Users/greg/Databases/MAIVb2013_adult_RAW.csv","rb") as f:
    for lcount,(l,s) in enumerate(zip(f.readlines(),subjects)):
        image_fname = l.split(",")[0]
        gold_string = l.split("\"")[1]
        gold_markings = gold_string[:-2].split(";")
        pts = [m.split(",")[:2] for m in gold_markings]
        pts = [(int(x),int(y)) for (x,y) in pts]
        print pts

        zooniverse_id = s["zooniverse_id"]
        print "----"
        if lcount == 3:
            break
