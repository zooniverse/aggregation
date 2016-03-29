#!/usr/bin/env python
from __future__ import print_function
import pymongo
import sys
sys.path.append("/home/ggdhines/github/aggregation/engine")
sys.path.append("/home/ggdhines/Pycharm/reduction/engine")
from agglomerative import Agglomerative
import json

client = pymongo.MongoClient()
db = client['penguin']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

# for c in classification_collection.find():
#     _id = c["_id"]
#     zooniverse_id = c["subjects"][0]["zooniverse_id"]
#
#     classification_collection.update_one({"_id":_id},{"$set":{"zooniverse_id":zooniverse_id}})

clustering_engine = Agglomerative(None,{})

# result = db.profiles.create_index([('zooniverse_id', pymongo.ASCENDING)],unique=False)
# print result
for c in classification_collection.find().limit(10):
    _id = c["_id"]
    zooniverse_id = c["subjects"][0]["zooniverse_id"]
    print(zooniverse_id)

    markings = []
    user_ids = []
    tools = []

    non_logged_in_users = 0
    for c2 in classification_collection.find({"zooniverse_id":zooniverse_id}):
        if "finished_at" in c2["annotations"][1]:
            continue

        if "user_name" in c2:
            id_ = c2["user_name"]
        else:
            id_ = c2["user_ip"]

        try:
            for penguin in c2["annotations"][1]["value"].values():
                x = float(penguin["x"])
                y = float(penguin["y"])
                penguin_type = penguin["value"]

                markings.append((x,y))
                user_ids.append(id_)
                tools.append(penguin_type)
        except AttributeError:
            continue

    if markings != []:
        clustering_results = clustering_engine.__cluster__(markings,user_ids,tools,markings,None,None)
        print(json.dumps(clustering_results[0][0], sort_keys=True,indent=4, separators=(',', ': ')))

        break