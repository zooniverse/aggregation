#!/usr/bin/env python
__author__ = 'greg'
from cassandra.cluster import Cluster
import cassandra
import pymongo
import uuid
import json
from cassandra.concurrent import execute_concurrent

cluster = Cluster()
cassandra_session = cluster.connect('serengeti')

# try:
#     cassandra_session.execute("drop table classifications")
#     print "table dropped"
# except cassandra.InvalidRequest:
#     print "table did not exist"
#     pass
# cassandra_session.execute("CREATE TABLE classifications(id int, created_at timestamp,zooniverse_id text,annotations text,user_name text, user_ip inet, PRIMARY KEY(id, created_at,user_ip)) WITH CLUSTERING ORDER BY (created_at ASC, user_ip ASC);")
cassandra_session.execute("CREATE TABLE ip_classifications (id int, created_at timestamp,zooniverse_id text,annotations text,user_name text, user_ip inet, PRIMARY KEY(id, user_ip,created_at)) WITH CLUSTERING ORDER BY (user_ip ASC,created_at ASC);")

# connect to the mongo server
client = pymongo.MongoClient()
db = client['serengeti_2015-02-22']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

insert_statement = cassandra_session.prepare("""insert into ip_classifications (id,created_at, zooniverse_id,annotations, user_name,user_ip)
                values (?,?,?,?,?,?)""")

statements_and_params = []

for ii,classification in enumerate(classification_collection.find()):
    created_at = classification["created_at"]
    if "user_name" in classification:
        user_name = classification["user_name"]
    else:
        user_name  = ""

    user_ip = classification["user_ip"]

    annotations = classification["annotations"]
    id = uuid.uuid1()
    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    params = (1,created_at,zooniverse_id,json.dumps(annotations),user_name,user_ip)

    statements_and_params.append((insert_statement, params))

    if (ii > 0) and (ii % 50000 == 0):
        print ii
        r = execute_concurrent(cassandra_session, statements_and_params, raise_on_first_error=True)
        statements_and_params = []

r = execute_concurrent(cassandra_session, statements_and_params, raise_on_first_error=True)