#!/usr/bin/env python
__author__ = 'greghines'
import pymongo
from datetime import datetime
from geoip import geolite2

client = pymongo.MongoClient()
db = client['galaxy_zoo_2015-01-01']
classification_collection = db["galaxy_zoo_classifications"]
user_collection = db["galaxy_zoo_users"]

for lang in ["en", "es", "fa", "it", "ru", "uk", "pt", "pl", "hu", "zh_tw", "zh_cn"]:
    print lang
    start = datetime(2014,6,1)
    print classification_collection.find({"annotations.lang":lang,"created_at":{"$gte":start}}).count()

    users = set([])
    ips = set([])
    for classification in classification_collection.find({"annotations.lang":lang,"created_at":{"$gte":start}}):
        try:
            users.add(classification["user_name"])
        except KeyError:
            continue

        try:
            ips.add(classification["user_ip"])
        except KeyError:
            continue

    countries = {}
    print len(list(users))
    print len(list(ips))
    total = 0
    for i in ips:

        match = geolite2.lookup(i)
        if match is not None:
            total += 1
            c = match.country
            if not c in countries:
                countries[c] = 1
            else:
                countries[c] += 1

    print total
    print countries
    print "===---"