#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import random
import os
import time
from time import mktime
from datetime import datetime,timedelta
import numpy as np
from scipy.stats import ks_2samp
import cPickle as pickle

project = "penguin"
date = "2015-02-22"

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

print