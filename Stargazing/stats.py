#!/usr/bin/env python
__author__ = 'greg'
from postgres_aggregation import PanoptesAPI
import panoptesPythonAPI
import os
import yaml
import urllib2
import requests
import json

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

if __name__ == "__main__":
    stargazing = PanoptesAPI("c")
    a,count = stargazing.__get_stats__()
    #print count
    #environment = os.getenv('ENVIRONMENT', "staging")

    # try:
    #     panoptes_file = open("config/aggregation.yml","rb")
    # except IOError:
    #     panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")
    # api_details = yaml.load(panoptes_file)
    #
    # userid = api_details[environment]["name"]
    # password = api_details[environment]["password"]


    # https://panoptes-comments.firebaseio.com/stargazing2015-zooniverse-org/projects/1/volunteers-count%22
    # host = "https://panoptes-comments.firebaseio.com/stargazing2015-zooniverse-org/projects/1/"

    # for i in range(20):
    #     try:
    #         http_api = panoptesPythonAPI.PanoptesAPI(host,userid,password,"")
    #         break
    #     except (urllib2.HTTPError,urllib2.URLError) as e:
    #         print "trying to connect/init again again"
    #         pass
    #print a
    #print
    response = requests.put("https://panoptes-comments.firebaseio.com/stargazing2015-zooniverse-org/projects/2/volunteers-count.json",data=str(a))
    response = requests.put("https://panoptes-comments.firebaseio.com/stargazing2015-zooniverse-org/projects/2/classifications-count.json",data=str(count))
    #print response