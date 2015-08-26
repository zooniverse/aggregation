#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import math
# from pymongo.objectid import ObjectId
from bson.objectid import ObjectId

# connect to the mongo server
client = pymongo.MongoClient()
db = client['oldWeather3-production-live']
# classification_collection = db["serengeti_classifications"]
# subject_collection = db["serengeti_subjects"]
# user_collection = db["serengeti_users"]
annotations = db["annotations"]
transcriptions = db["transcriptions"]
assets = db["assets"]
voyages = db["voyages"]
ships = db["ships"]

bear_ship = ships.find_one({"name":"Bear"})
voy = voyages.find_one({"ship_id":bear_ship["_id"]})



# for page in assets.find({"voyage_id":voy["_id"]}):
#     print page["location"]

# for ann in annotations.find({"asset_id":"ObjectId('500ddfffd2ca730755000633')"}):
for ann in annotations.find():#{"asset_id":"ObjectId('500ddfffd2ca730755000633')"}):
    transcription_id = ann["transcription_id"]

    transcription = transcriptions.find_one({"_id":ObjectId(transcription_id)})
    asset = assets.find_one({"_id":ObjectId(transcription["asset_id"])})
    location = asset["location"]
    if ("Bear" in location) and ("72" in location):
        print ann["data"]

        print
        # print transcription
        # print location
        # break

# print annotations.find_one({'transcription_id': ObjectId("500e0f6ad2ca73074b00089b")})
#
# print transcriptions.find_one()