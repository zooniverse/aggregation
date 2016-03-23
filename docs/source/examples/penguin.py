import pymongo

client = pymongo.MongoClient()
db = client['penguin']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

# for c in classification_collection.find():
#     _id = c["_id"]
#     zooniverse_id = c["subjects"][0]["zooniverse_id"]
#
#     classification_collection.update_one({"_id":_id},{"$set":{"zooniverse_id":zooniverse_id}})


# result = db.profiles.create_index([('zooniverse_id', pymongo.ASCENDING)],unique=False)

# for c in classification_collection.find().limit(10):
#     _id = c["_id"]
#     zooniverse_id = c["subjects"][0]["zooniverse_id"]
#
#     for c2 in classification_collection.find({"zooniverse_id":zooniverse_id}):
#         print c2


import sys
sys.path.append("/home/ggdhines/github/aggregation/engine")
from agglomerative import Agglomerative