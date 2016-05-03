#!/usr/bin/env python
import pymongo
import psycopg2
import json

client = pymongo.MongoClient()
db = client['penguin']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

# conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='apassword'")
# conn.autocommit = True
# cur = conn.cursor()
# cur.execute("create database penguins")

conn = psycopg2.connect("dbname='penguins' user='postgres' host='localhost' password='apassword'")
conn.autocommit = True
cur = conn.cursor()
# cur.execute("drop table classifications")
# cur.execute("create table classifications (zooniverse_id text, user_id text, annotations json, PRIMARY KEY(zooniverse_id, user_id))")
cur.execute("create index ids_ on classifications (zooniverse_id ASC)")

# for ii,classification in enumerate(classification_collection.find()):
#
#     zooniverse_id = classification["subjects"][0]["zooniverse_id"]
#     if ii % 100 == 0:
#         print(ii)
#
#     if "user_name" in classification:
#         id_ = classification["user_name"]
#         id_ = id_.encode('ascii','ignore')
#         id_ = id_.replace("'","")
#     else:
#         id_ = classification["user_ip"]
#
#     if "finished_at" in classification["annotations"][1]:
#         continue
#
#     annotations = json.dumps(classification["annotations"])
#     # annotations = annotations.replace("\"","")
#     annotations = annotations.replace("'","")
#     try:
#         cur.execute("insert into classifications values ('"+str(zooniverse_id)+"','"+str(id_)+"','"+annotations + "')")
#     except psycopg2.IntegrityError as e:
#         pass
#
# conn.commit()
