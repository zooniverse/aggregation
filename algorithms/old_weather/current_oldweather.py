#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import math
import matplotlib.cbook as cbook
# from pymongo.objectid import ObjectId
from bson.objectid import ObjectId
import os
import urllib

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"


def image_setup(url):

    slash_index = url.rfind("/")
    fname = url[slash_index+1:]

    image_path = base_directory+"/Databases/images/"+fname

    if not(os.path.isfile(image_path)):
        urllib.urlretrieve(url, image_path)

    return image_path

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
        if "wind_force" in ann["data"]:
            # print ann["data"]
            # print ann
            # print asset
            # break
            # print
            x = transcription["page_data"]["asset_screen_width"]
            y = transcription["page_data"]["asset_screen_height"]

            print ann
            print transcription

            # print ann["bounds"]


            fname = image_setup(location)
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            image_file = cbook.get_sample_data(fname)
            image = plt.imread(image_file)

            print image.shape
            plt.plot(x,y,"o")

            # fig, ax = plt.subplots()
            im = axes.imshow(image)
            plt.show()
            break

# print annotations.find_one({'transcription_id': ObjectId("500e0f6ad2ca73074b00089b")})
#
# print transcriptions.find_one()