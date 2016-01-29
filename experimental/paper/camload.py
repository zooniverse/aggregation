#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import random
import os
import urllib2
import cPickle as pickle

project = "serengeti"
date = "2014-07-28"

# for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
    code_directory = base_directory + "/github"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"

else:
    base_directory = "/home/greg"
    code_directory = base_directory + "/github"

client = pymongo.MongoClient()
db = client[project+"_"+date]
classification_collection = db[project+"_classifications"]
subject_collection = db[project+"_subjects"]
user_collection = db[project+"_users"]

complete_subjects = list(subject_collection.find({"state":"complete", "tutorial":{"$ne":True},"metadata.counters.blank":{"$gte":5}}).limit(300000))

to_process = random.sample(complete_subjects,250)
pickle.dump(to_process,open(base_directory+"/Databases/serengeti/blank.pickle","wb"))
for subject in to_process:
    url_l = subject["location"]["standard"]

    try:
        slash_index = url_l[0].rfind("/")
        object_id = url_l[0][slash_index+1:]
    except IndexError:
        print url_l
        continue

        #print object_id

    if not(os.path.isfile(base_directory+"/Databases/"+project+"/images/blank/"+object_id)):
        #urllib.urlretrieve(url, base_directory+"/Databases/"+project+"/images/"+object_id)
        imgRequest = urllib2.Request(url_l[0])
        imgData = urllib2.urlopen(imgRequest).read()
        output = open( base_directory+"/Databases/"+project+"/images/blank/"+object_id,'wb')
        output.write(imgData)
        output.close()

# complete_subjects = list(subject_collection.find({"state":"complete", "tutorial":{"$ne":True},"metadata.counters.blank":{"$lte":4}}).limit(300000))
#
# to_process = random.sample(complete_subjects,250)
# pickle.dump(to_process,open(base_directory+"/Databases/serengeti/not_blank.pickle","wb"))
# for subject in to_process:
#     url_l = subject["location"]["standard"]
#     try:
#         slash_index = url_l[0].rfind("/")
#         object_id = url_l[0][slash_index+1:]
#     except IndexError:
#         print url_l
#         continue
#
#     #print object_id
#
#     if not(os.path.isfile(base_directory+"/Databases/"+project+"/images/notblank/"+object_id)):
#         #urllib.urlretrieve(url, base_directory+"/Databases/"+project+"/images/"+object_id)
#         imgRequest = urllib2.Request(url_l[0])
#         imgData = urllib2.urlopen(imgRequest).read()
#         output = open( base_directory+"/Databases/"+project+"/images/notblank/"+object_id,'wb')
#         output.write(imgData)
#         output.close()