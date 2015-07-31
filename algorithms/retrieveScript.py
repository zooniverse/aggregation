#!/usr/bin/env python

__author__ = 'ggdhines'
from penguin import Penguins
import json
project = Penguins()

stmt = "select subject_id,aggregation from aggregations where workflow_id = -1"
cursor = project.postgres_session.cursor()

# cursor.execute(stmt)
# for i in cursor.fetchall():
#     path = None
#     aggregations = i[1]#json.loads(i[1])
#
#     for cluster_index,cluster in  aggregations["1"]["point clusters"].items():
#         if cluster_index in ["param","all_users"]:
#             continue
#
#         # print cluster["existence"][0]
#         if isinstance(cluster["existence"][0],dict):
#             p = cluster["existence"][0]["1"]
#         else:
#
#             p = cluster["existence"][0][1]
#
#         if p > 0.5:
#             if path is None:
#                 subject = project.subject_collection.find_one({"zooniverse_id":i[0]})
#                 path = subject["metadata"]["path"][:-4]
#                 url = subject["location"]["standard"]
#
#             print str(path) +"," + str(cluster["center"])  + "," + str(url)
#
#     break


for subject in project.subject_collection.find():
    print subject["zooniverse_id"] + "," + subject["metadata"]["path"][:-4]
