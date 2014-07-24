#!/usr/bin/env python
import pymongo
__author__ = 'greghines'

# AMW0000v75
# AMW0000tf7
# AMW0000rvj
# AMW0000qwf
# AMW0000fu3
# AMW0000ieg
# AMW0000oll
# AMW0000fo1
# AMW000079z
# AMW00007wk
# AMW0000p96
# AMW0000u6f
# AMW0000puo
# AMW0000qmu
# AMW0000thg
# AMW0000uey
# AMW0000u2n
# AMW0000tz7
# AMW0000k6s
# AMW0000t55
# AMW0000jec
# AMW0000u7h
# AMW0000u6e
# AMW0000poo
# AMW0000nqo
# AMW0000guf
# AMW0000uc5
# AMW0000t4p
# AMW0000po2
# AMW0000px9
# AMW0000v8d
# AMW0000p8e
# AMW0000pu1
# AMW0000v1f
# AMW0000vah

goldStandardIDs = []
enoughData = []

#load gold standard data
client = pymongo.MongoClient()
db = client['milky_way']
collection = db["milky_way_classifications"]

for classification in collection.find({"$or" : [{"user_name":"ttfnrob"}]}):
    subject_zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    if not(subject_zooniverse_id in goldStandardIDs):
        goldStandardIDs.append(subject_zooniverse_id)

for subjectID in goldStandardIDs:
    count = 0
    for classification in collection.find({"subjects.zooniverse_id":subjectID, "user_name": {"$nin":["stuart.lynn","ttfnrob"]}}):

        #assert classification["user_name"] != "stuart.lynn"
        count += 1

    if count >= 50:
        enoughData.append(subjectID)

print enoughData

