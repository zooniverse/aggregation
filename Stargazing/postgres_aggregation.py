#!/usr/bin/env python
__author__ = 'greg'
import math
import panoptesPythonAPI
import time
import yaml
import psycopg2

# calculate the score associated with a given classification according to the algorithm
# in the paper Galaxy Zoo Supernovae
# an example of the json format used is
# [{u'task': u'centered_in_crosshairs', u'value': 1}, {u'task': u'subtracted', u'value': 1}, {u'task': u'circular', u'value': 1}, {u'task': u'centered_in_host', u'value': 0}]
def score_index(annotations):
    assert annotations[0]["task"] == "centered_in_crosshairs"
    if annotations[0]["value"] == 0:
        return 0  #-1

    # they should have answered yes
    assert annotations[1]["task"] == "subtracted"
    if annotations[1]["value"] == 0:
        return 0  #-1

    assert annotations[2]["task"] == "circular"
    if annotations[2]["value"] == 0:
        return 0  #-1

    assert annotations[3]["task"] == "centered_in_host"
    if annotations[3]["value"] == 0:
        return 2  #3
    else:
        return 1  #1

# get my userID and password
# purely for testing, if this file does not exist, try opening on Greg's computer
try:
    panoptes_file = open("config/aggregation.yml","rb")
except IOError:
    panoptes_file = open("/home/greg/Databases/aggregation.yml","rb")
login_details = yaml.load(panoptes_file)
userid = login_details["name"]
password = login_details["password"]

# get the token necessary to connect with panoptes
token = panoptesPythonAPI.get_bearer_token(userid,password)

# get the project id for Supernovae and the workflow version
project_id = panoptesPythonAPI.get_project_id("Supernovae",token)
workflow_version = panoptesPythonAPI.get_workflow_version(project_id,token)
workflow_id = panoptesPythonAPI.get_workflow_id(project_id,token)

print "project id  is " + str(project_id)
print "workflow id is " + str(workflow_id)

# now load in the details for accessing the database
try:
    database_file = open("config/database.yml")
except IOError:
    database_file = open("/home/greg/Databases/database.yml")
database_details = yaml.load(database_file)

environment = "staging"
database = database_details[environment]["database"]
username = database_details[environment]["username"]
password = database_details[environment]["password"]
host = database_details[environment]["host"]

# try connecting to the db
try:
    details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
    conn = psycopg2.connect(details)
except:
    print "I am unable to connect to the database"

def find_subjects():
    cur = conn.cursor()
    #cur.execute("Select * from subject_sets where ")

    # cur.execute("SELECT subject_id from set_member_subjects where subject_set_id=4 OR subject_set_id=3")
    # cur.execute("SELECT expert_set from subject_sets where subject_sets.project_id="+str(project_id)+" and subject_sets.workflow_id=" + str(workflow_id))
    # rows = cur.fetchall()
    #
    # for r in rows:
    #     print r


    cur.execute("SELECT subject_id from set_member_subjects inner join subject_sets on set_member_subjects.subject_set_id=subject_sets.id where (subject_sets.expert_set = FALSE or subject_sets.expert_set IS NULL) and subject_sets.project_id="+str(project_id)+" and subject_sets.workflow_id=" + str(workflow_id))
    rows = cur.fetchall()

    return [r[0] for r in rows]

def calc_scores_(subject_id):
    scores = {}
    cur = conn.cursor()
    cur.execute("SELECT annotations from classifications where project_id="+str(project_id)+" and workflow_id=" + str(workflow_id) + " and subject_ids = ARRAY["+str(subject_id)+"]")
    rows = cur.fetchall()

    for subject_count,annotations in enumerate(rows):
        #print annotations
        if not((workflow_id,subject_id) in scores):
            scores[(workflow_id,subject_id)] = [0,0,0]
        scores[(workflow_id,subject_id)][score_index(annotations[0])] += 1
    print "read in " + str(subject_count) + " classifications"
    return scores

def calc_scores(scores = {}):
    cur = conn.cursor()
    cur.execute("SELECT subject_ids,annotations from classifications where project_id="+str(project_id)+" and workflow_id=" + str(workflow_id))
    #cur.execute("SELECT * from classifications where project_id="+str(project_id)+" and workflow_id=" + str(workflow_id))
    rows = cur.fetchall()

    # for r in rows:
    #     print r
    #
    # return

    for subject_count,(subjects,annotations) in enumerate(rows):
        #print annotations
        subject_id = subjects[0]
        if not((workflow_id,subject_id) in scores):
            scores[(workflow_id,subject_id)] = [0,0,0]
        scores[(workflow_id,subject_id)][score_index(annotations)] += 1
    #print "read in " + str(subject_count) + " classifications"
    return scores

def aggregate(scores):
    # now do the aggregation
    for (workflow_id,subject_id),values in scores.items():
        # calculate the average score, std and set up the JSON results
        avg_score = (values[0]*-1+ + values[1]*1 + values[2]*3)/float(sum(values))
        std = math.sqrt((-1-avg_score)**2*(values[0]/float(sum(values))) + (1-avg_score)**2*(values[1]/float(sum(values))) + (3-avg_score)**2*(values[1]/float(sum(values))))
        aggregation = {"mean":avg_score,"std":std,"count":values}

        # try creating the aggregation
        status,explanation = panoptesPythonAPI.create_aggregation(workflow_id,subject_id,token,aggregation)

        # if we had a problem, try updating the aggregation
        if status == 400:
            aggregation_id,etag = panoptesPythonAPI.find_aggregation_etag(workflow_id,subject_id,token)
            panoptesPythonAPI.update_aggregation(workflow_id,workflow_version,subject_id,aggregation_id,token,aggregation,etag)

if __name__ == "__main__":
    scores = calc_scores()
    aggregate(scores)
    # for subject_id in subjects:
    # calc_scores_(5)
    # for i in range(1):
    #     print i
    #     start = time.time()
    #     scores = calc_scores(scores)
    #     end = time.time()
    #
    #     print end - start
    #
    # aggregate(scores)

