#!/usr/bin/env python
__author__ = 'greg'
import cassandra
from cassandra.cluster import Cluster
import json
import aggregation_api

# connect to the cassandra node
cluster = Cluster(['panoptes-cassandra.zooniverse.org'])

# try to connect to the zooniverse keyspace - if it doesn't exist, create it
try:
    session = cluster.connect('zooniverse')
except cassandra.InvalidRequest:
    session = cluster.connect()
    session.execute("CREATE KEYSPACE zooniverse WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }")
    session = cluster.connect('zooniverse')

# create the classifications table
# uncomment below if you want to delete it in case it already exists
session.execute("drop table classifications")
# if the table already exists, just continue
try:
    session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, subject_id int, workflow_version float,metadata text, PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
except cassandra.AlreadyExists:
    pass


# create the aggregation table
try:
    session.execute("drop table aggregations")
except cassandra.InvalidRequest as e:
    print e
    pass

try:
    session.execute("CREATE TABLE aggregations (subject_id int, workflow_id int, task text, aggregation text, created_at timestamp, updated_at timestamp, metadata text, PRIMARY KEY(subject_id,workflow_id) ) WITH CLUSTERING ORDER BY (workflow_id ASC);")
except cassandra.InvalidRequest as e:
    print e
    pass
# # self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, metadata text, subject_id int, workflow_version text, PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
#
#
# # session.execute("CREATE KEYSPACE demo WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }")
#
#
#
p = aggregation_api.PanoptesAPI("bar_lengths")
p.__migrate__()

# def __load__():
#
#
#     project_id = p.project_id
#     workflow_version = p.workflow_version
#     workflow_id = p.workflow_id
#
#     # session.execute("drop table classifications")
#     # session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed bool, gold_standard bool, metadata text, subject_id int, workflow_version int PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
#
#     for ii,t in enumerate(p.__yield_classifications__(limit=100000)):
#         #"SELECT user_id,created_at,user_ip,annotations,subject_ids from classifications
#         id_ = t[0]
#         user_id = t[1]
#         if user_id is None:
#             user_id = -1
#         created_at = t[2]
#         user_ip = t[3]
#         annotations = json.dumps(t[4])
#         subject_ids = t[5]
#         classification = [int(project_id),subject_ids[0],user_id,annotations,created_at,user_ip]
#         print classification
#         # session.execute("INSERT INTO classifications (project_id, subject_ids, user_id, annotations, created_at, user_ip, workflow_version, workflow_id,  uid) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",classification)
#         session.execute("INSERT INTO classifications (project_id, subject_id, user_id, annotations, created_at, user_ip) VALUES (%s,%s,%s,%s,%s,%s)",classification)
#
# __load__()

# def __analyze__():
#     rows = session.execute("SELECT * from classifications")
#     for r in rows:
#         print r.subject_id
#         print json.loads(r.annotations)
#
#         break
#
# # __load__()