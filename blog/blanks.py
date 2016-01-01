#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import urllib
import matplotlib.pyplot as plt
import cv2
from cassandra.cluster import Cluster
import cassandra
from cassandra.concurrent import execute_concurrent
import psycopg2
import psycopg2
import numpy as np
from scipy.stats import chi2
import math

db_name = "serengeti"
date = "_2015-08-20"

client = pymongo.MongoClient()
db = client[db_name+date]
subjects = db[db_name+"_subjects"]

for i in subjects.find({"tutorial":{"$ne":True},u'coords': [-2.4672743413359295, 34.75278520232197],"metadata.retire_reason":"blank"}).limit(30):
    files = []
    for url in i["location"]["standard"]:
        assert isinstance(url,unicode)
        index = url.rfind("/")
        fname = url[index+1:]
        image_path = "/home/ggdhines/Databases/images/"+fname

        files.append(image_path)

        if not(os.path.isfile(image_path)):
            url = "http://zooniverse-static.s3.amazonaws.com/" + url[7:]
            urllib.urlretrieve(url, image_path)

    break