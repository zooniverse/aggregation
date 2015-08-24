#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import math

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



for page in assets.find({"voyage_id":voy["_id"]}):
    print page["location"]

for ann in annotations.find({"asset_id":"500ddfffd2ca730755000633"}):
    print ann



