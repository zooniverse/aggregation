#!/usr/bin/env python
__author__ = 'ggdhines'
import os
from postgres_aggregation import PanoptesAPI
from cassandra.cluster import Cluster
from cassandra.cqltypes import UUIDType

# for Greg running on either office/home - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

cluster = Cluster()
session = cluster.connect('panoptes')
results = session.execute("TRUNCATE classification_model")

prepared_stmt = session.prepare("INSERT INTO classification_model (user_id,created_at,user_ip,annotations,subject_id, classification_id) VALUES (?, ?, ?, ?, ?, uuid())")

stargazing = PanoptesAPI()
stargazing.__cleanup__()

for ii,t in enumerate(stargazing.__yield_classifications__()):
    # convert subjects_ids into subject_id
    t = list(t)
    t[-1] = t[-1][0]

    bound_stmt = prepared_stmt.bind(t)
    stmt = session.execute(bound_stmt)
    if (ii%1000) == 0:
        print ii
