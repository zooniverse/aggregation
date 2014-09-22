#!/usr/bin/env python
import sys
from sklearn.cluster import DBSCAN
import numpy as np
import datetime
import param

def isBlank(votes):
    v = votes.sort(key = lambda x:x[1])
    if not(0 in v[:classifications_for_blank]):
        return True

#read in config file
configFile = sys.argv[1]
project_type = None
with open(configFile, 'r') as conf:
    configuration = conf.read()
    exec(configuration)

if project_type == "animalMarking":
    mainIndex = sorted(param_list.keys()).index("animal")

    main_index = sorted(param.required_param["animalMarking"].keys()).index("animal")

time_index = sorted(param.required_param[project_type].keys()).index("time_stamp")
user_index = sorted(param.required_param[project_type].keys()).index("user_name")

curr_subject = None
blank_classifications = {}
# input comes from STDIN (standard input)
i = 0
for line in sys.stdin:

    subject_id,v = line[:-1].split("\t")

    if curr_subject != subject_id:
        if curr_subject is not None:
            print blank_classifications.values()
        curr_subject = subject_id
        blank_classifications = {}


    param = v.split(",")
    #if we are in production, we don't care about the time stamps
    if production:
        time_stamp = 0
    else:
        time_stamp = datetime.datetime.strptime(param[time_index], "%Y-%m-%d %H:%M:%S %Z")
    user_id = param[1]

    if (param[main_index] in blank_types) and not(user_id in blank_classifications):
        blank_classifications[user_id] = (1,time_stamp)
    else:
        blank_classifications[user_id] = (0,time_stamp)


if curr_subject is not None:
    print blank_classifications.values()
