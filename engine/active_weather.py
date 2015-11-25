import aggregation_api
from cassandra.cluster import Cluster

cluster = Cluster()
cassandra_session = cluster.connect("active_weather")

project_id = 1335
workflow_id = 1915

project = aggregation_api.AggregationAPI(project_id,"staging")
# project.__migrate__()

subject_set = project.__get_subjects__(workflow_id,only_retired_subjects=False)

transcriptions,_,dimensions = project.__sort_annotations__(workflow_id,subject_set)

print dimensions

columns = [(a.lb,a.ub) for a in cassandra_session.execute("select * from columns")]
columns.sort(key = lambda x:x[0])

rows = [(a.lb,a.ub) for a in cassandra_session.execute("select * from rows")]
rows.sort(key = lambda x:x[0])

for subject_id,markings in transcriptions["T2_0_0"].items():
    f_name = project.__get_subject_metadata__(subject_id)["subjects"][0]["metadata"]["Filename"]

    print f_name

    for (pos,user_id),t in markings.items():
        print pos,t

        found = False

        print columns
        for column_index,(lb,ub) in enumerate(columns):
            if (lb <= pos[0]) and (pos[0] <= ub):
                found = True
                break

        print found