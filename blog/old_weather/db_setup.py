from cassandra.cluster import Cluster
import cassandra

cluster = Cluster()

try:
    cassandra_session = cluster.connect("active_weather")
except cassandra.InvalidRequest:
    cassandra_session = cluster.connect()
    cassandra_session.execute("CREATE KEYSPACE active_weather WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 2 }")
    cassandra_session = cluster.connect("active_weather")

cassandra_session.execute("DROP TABLE columns")
cassandra_session.execute("DROP TABLE rows")

try:
    cassandra_session.execute("CREATE TABLE columns (lb int, ub int, PRIMARY KEY(lb,ub))")
except cassandra.AlreadyExists:
    pass

try:
    cassandra_session.execute("CREATE TABLE rows (lb int, ub int, PRIMARY KEY(lb,ub))")
except cassandra.AlreadyExists:
    pass


cell_columns = [(753,855),(855,923),(923,1024),(1024,1092),(1432,1532),(1532,1734),(1734,1827),(1827,1934),(1934,2035),(2035,2133),(2707,2832),(2832,2935),(2935,3072)]
cell_rows = [(1295,1394),(1394,1447),(1447,1501),(1501,1558),(1558,1614)]

for lb,ub in cell_columns:
    cassandra_session.execute("insert into columns (lb, ub) VALUES (%s,%s)",(lb,ub))

for lb,ub in cell_rows:
    cassandra_session.execute("insert into rows (lb, ub) VALUES (%s,%s)",(lb,ub))

columns = [(a.lb,a.ub) for a in cassandra_session.execute("select * from columns")]
columns.sort(key = lambda x:x[0])

print columns


