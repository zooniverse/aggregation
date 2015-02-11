#!/usr/bin/env python
__author__ = 'greg'
import cPickle as pickle
import os
import unirest

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

to_process = pickle.load(open(base_directory+"/Databases/serengeti/blank.pickle","rb"))

blank_images = {}

for id,subject in enumerate(to_process):
    print id
    if id == 50:
        break
    url_l = subject["location"]["standard"]

    try:
        slash_index = url_l[0].rfind("/")
        object_id = url_l[0][slash_index+1:]
    except IndexError:
        print url_l
        continue


    # These code snippets use an open-source library. http://unirest.io/python
    response = unirest.post("https://camfind.p.mashape.com/image_requests",
      headers={
        "X-Mashape-Key": "oxzr8wVhormshATe9Ny2hLeuGFXKp1kowXujsn6hycC1PbXzLx",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
      },
      params={
        "focus[x]": "480",
        "focus[y]": "640",
        "image_request[altitude]": "27.912109375",
        "image_request[language]": "en",
        "image_request[latitude]": "35.8714220766008",
        "image_request[locale]": "en_US",
        "image_request[longitude]": "14.3583203002251",
        "image_request[remote_image_url]": url_l[0]
      }
    )

    blank_images[subject["zooniverse_id"]] = response.body["token"]

print "====---"

pickle.dump(blank_images,open(base_directory+"/Databases/serengeti/blank_dict.picke","wb"))

######
######

to_process = pickle.load(open(base_directory+"/Databases/serengeti/not_blank.pickle","rb"))

not_blank_images = {}

for id,subject in enumerate(to_process):
    print id
    url_l = subject["location"]["standard"]

    try:
        slash_index = url_l[0].rfind("/")
        object_id = url_l[0][slash_index+1:]
    except IndexError:
        print url_l
        continue


    # These code snippets use an open-source library. http://unirest.io/python
    response = unirest.post("https://camfind.p.mashape.com/image_requests",
      headers={
        "X-Mashape-Key": "oxzr8wVhormshATe9Ny2hLeuGFXKp1kowXujsn6hycC1PbXzLx",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
      },
      params={
        "focus[x]": "480",
        "focus[y]": "640",
        "image_request[altitude]": "27.912109375",
        "image_request[language]": "en",
        "image_request[latitude]": "35.8714220766008",
        "image_request[locale]": "en_US",
        "image_request[longitude]": "14.3583203002251",
        "image_request[remote_image_url]": url_l[0]
      }
    )

    not_blank_images[subject["zooniverse_id"]] = response.body["token"]

print "====---"

pickle.dump(not_blank_images,open(base_directory+"/Databases/serengeti/not_blank_dict.picke","wb"))