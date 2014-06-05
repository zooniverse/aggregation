#!/usr/bin/env python
from __future__ import print_function
import pymongo
import os
from subprocess import call
from PIL import Image
from dateutil import parser
import ephem
__author__ = 'ggdhines'

size = 128, 128

os.chdir("/home/ggdhines/Databases/serengeti/photos/")

client = pymongo.MongoClient()
db = client['serengeti_2014-05-13']
collection = db['serengeti_subjects']

blank_photos = []
photo_names = []
daylight_photos = []

o=ephem.Observer()
o.lat='-2.4672743413359295'
o.lon='34.75278520232197'


for document in collection.find({"coords": [-2.4672743413359295, 34.75278520232197]})[0:300]:
    photo = document["location"]["standard"][0]



    i = photo.rfind("/")
    i2 = photo.rfind(".")
    image_id = str(photo[i+1:i2])
    photo_names.append(image_id+".thumbnail")

    #when was this photo taken?
    time = document["metadata"]["timestamps"][0]
    #convert time to UTC
    newHour = time.hour - 3
    newDay = time.day
    if newHour < 0:
        newHour += 24
        newDay += -1


    try:
        newTime = time.replace(day=newDay,hour=newHour)
    except ValueError:
        print("====error")
        continue



    if document["metadata"]["retire_reason"] == "blank":
        blank_photos.append(True)
    else:
        blank_photos.append(False)

    if not(os.path.isfile(image_id+".jpg")):
        #urllib.urlretrieve(p,baseDir+image_id)
        call(["aws", "s3", "cp", "s3://www.snapshotserengeti.org/subjects/standard/"+image_id+".jpg", "."])

    #have we already created the thumbnail
    if not(os.path.isfile(image_id+".thumbnail")):
        im = Image.open(image_id+".jpg")
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(image_id + ".thumbnail", "JPEG")

    continue
    #########
    #time stuff
    #get the sunrise and sunset
    o.date = str(time.year)+"-"+str(time.month)+"-"+str(time.day)+" 9:00:00"
    sunriseStr = str(o.previous_rising(ephem.Sun()))
    sunrise = parser.parse(sunriseStr)

    sunsetStr = str(o.next_setting(ephem.Sun()))
    sunset = parser.parse(sunsetStr)
    if ((newTime >= sunrise) and (newTime <= sunset)):
        #print("day")
        call(["cp",image_id+".thumbnail","day/"])
    else:
        #print("night")
        call(["cp",image_id+".thumbnail","night/"])

print(sum(blank_photos))