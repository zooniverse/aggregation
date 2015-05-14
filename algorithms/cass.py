#!/usr/bin/env python
__author__ = 'greg'
# CREATE TABLE classifications( project_id int, user_id int, annotations text, created_at timestamp, user_ip inet, workflow_version text, workflow_id int, subject_ids list<int>, uid uuid primary key );
# create index subjects on classifications(subject_ids);
# create index users on classifications(user_id);

from cassandra.cluster import Cluster
import uuid
import panoptesPythonAPI
import json

cluster = Cluster()
session = cluster.connect('demo')




def __load__():
    p = panoptesPythonAPI.PanoptesAPI()
    p.__postgres_connect__()

    project_id = p.project_id
    workflow_version = p.workflow_version
    workflow_id = p.workflow_id

    session.execute("drop table classifications")
    session.execute("CREATE TABLE classifications( project_id int, subject_id int, user_id int, annotations text, created_at timestamp, user_ip inet,  PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")

    for ii,t in enumerate(p.__yield_classifications__(limit=100000)):
        #"SELECT user_id,created_at,user_ip,annotations,subject_ids from classifications
        id_ = t[0]
        user_id = t[1]
        if user_id is None:
            user_id = -1
        created_at = t[2]
        user_ip = t[3]
        annotations = json.dumps(t[4])
        subject_ids = t[5]
        classification = [int(project_id),subject_ids[0],user_id,annotations,created_at,user_ip]
        print classification
        # session.execute("INSERT INTO classifications (project_id, subject_ids, user_id, annotations, created_at, user_ip, workflow_version, workflow_id,  uid) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",classification)
        session.execute("INSERT INTO classifications (project_id, subject_id, user_id, annotations, created_at, user_ip) VALUES (%s,%s,%s,%s,%s,%s)",classification)

def __analyze__():
    rows = session.execute("SELECT * from classifications")
    for r in rows:
        print r.subject_id
        print json.loads(r.annotations)

        break

__load__()