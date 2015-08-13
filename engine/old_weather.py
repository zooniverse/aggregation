__author__ = 'ggdhines'
import aggregation_api

project = aggregation_api.AggregationAPI(project = 195, environment="staging",cassandra_connection=False)

cursor = project.postgres_session.cursor()
# stmt = "select name from projects "
# cursor.execute(stmt)

# for r in cursor.fetchall():
#     if r[1] == None:
#         continue
#     if "old" in r[1]:
#         print r

stmt = "select * from workflows where project_id = 195"
cursor.execute(stmt)

for r in cursor.fetchall():
    print r[0]

assert False

# print project.workflows
# print project.__sort_annotations__()

stmt = "select subject_ids,annotations from classifications where workflow_id = 611"
cursor.execute(stmt)

for subject_ids,annotations  in cursor.fetchall():
    subject_id = subject_ids[0]
    print project.__image_setup__(subject_id)
    break