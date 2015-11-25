from cassandra.cluster import Cluster
import cassandra

cluster = Cluster()

try:
    cassandra_session = cluster.connect("active_weather")
except cassandra.InvalidRequest:
    cassandra_session = cluster.connect()
    cassandra_session.execute("CREATE KEYSPACE active_weather WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 2 }")
    cassandra_session = cluster.connect("active_weather")

try:
    cassandra_session.execute("CREATE TABLE columns (lb int, ub int, PRIMARY KEY(lb,ub))")
except cassandra.AlreadyExists:
    pass

try:
    cassandra_session.execute("CREATE TABLE rows (lb int, ub int, PRIMARY KEY(lb,ub))")
except cassandra.AlreadyExists:
    pass


cell_columns = [(510,713),(713,821),(821,890),(1219,1252),(1527,1739),(1739,1837),(1837,1949),(1949,2053),(2053,2156)]
cell_rows = [(1226,1320),(1320,1377)]

for lb,ub in cell_columns:
    cassandra_session.execute("insert into columns (lb, ub) VALUES (%s,%s)",(lb,ub))

for lb,ub in cell_rows:
    cassandra_session.execute("insert into rows (lb, ub) VALUES (%s,%s)",(lb,ub))

columns = [(a.lb,a.ub) for a in cassandra_session.execute("select * from columns")]
columns.sort(key = lambda x:x[0])

print columns


