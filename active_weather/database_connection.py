from cassandra.cluster import Cluster,InvalidRequest
import cPickle as pickle
__author__ = 'ggdhines'


class Database:
    def __init__(self):
        cluster = Cluster()

        try:
            self.cassandra_session = cluster.connect("active_weather")
        except InvalidRequest as e:
            print e
            # cassandra_session = cluster.connect()
            # cassandra_session.execute("CREATE KEYSPACE ActiveWeather WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 2 }")
            self.cassandra_session = cluster.connect('active_weather')

        columns = "id_ int, subject_id text, region int, row int, column int, character text, position int, bitmap text"

        primary_key = "subject_id,region,row,column,position"
        ordering = "region ASC,row ASC,column ASC, position ASC"

        self.cassandra_session.execute("drop table transcriptions")
        self.cassandra_session.execute("CREATE TABLE transcriptions(" + columns + ", PRIMARY KEY( " + primary_key + ")) WITH CLUSTERING ORDER BY ( " + ordering + ");")

    def __has_cell_been_transcribed__(self,subject_id,region,row,column):
        select_stmt = "select count(*) from transcriptions where subject_id = '"+ str(subject_id) + "'"
        select_stmt += " and region = " + str(region) + " and row = " + str(row) + " and column = " + str(column)

        results = self.cassandra_session.execute(select_stmt)

        return results != []

    def __add_transcription__(self,subject_id,region,row,column,position,character,bitmap):
        insert_stmt = "insert into transcriptions (subject_id,region,row,column,position,character,bitmap) values ("
        insert_stmt += "'" + subject_id + "'," + str(region) + "," + str(row) + "," + str(column) + "," +str(position)
        insert_stmt += ",'" + character + "','" + pickle.dumps(bitmap) + "')"

        self.cassandra_session.execute(insert_stmt)
