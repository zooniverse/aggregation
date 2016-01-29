#!/usr/bin/env python
__author__ = 'greg'
import os
import cPickle as pickle
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

not_blank_dict = pickle.load(open(base_directory+"/Databases/serengeti/not_blank_dict.picke","rb"))

results = {}

for zooniverse_id,token in not_blank_dict.items():
    print zooniverse_id
    # These code snippets use an open-source library. http://unirest.io/python
    response = unirest.get("https://camfind.p.mashape.com/image_responses/"+token,
      headers={
        "X-Mashape-Key": "oxzr8wVhormshATe9Ny2hLeuGFXKp1kowXujsn6hycC1PbXzLx",
        "Accept": "application/json"
      }
    )

    results[zooniverse_id] = response.body
    print response.body

pickle.dump(results,open(base_directory+"/Databases/serengeti/results.pickle","wb"))
