#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo

client = pymongo.MongoClient()
db = client['plankton_2015-01-01']
classification_collection = db["plankton_classifications"]
subject_collection = db["plankton_subjects"]
user_collection = db["plankton_users"]

error = 0
error2 = 0
for classification in classification_collection.find().sort("created_at",pymongo.DESCENDING).limit(50000):
    if not("started_at" in classification["annotations"][0]):
        try:
            if classification["annotations"][0]["species"] == "":
                error += 1
        except KeyError:
            print classification["annotations"][0]
            error2 += 1

print classification

print error,error2