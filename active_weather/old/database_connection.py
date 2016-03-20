from __future__ import print_function
from cassandra.cluster import Cluster,InvalidRequest
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
__author__ = 'ggdhines'


class Database:
    def __init__(self):
        cluster = Cluster(["panoptes-cassandra.zooniverse.org"])

        try:
            self.cassandra_session = cluster.connect("active_weather")
        except InvalidRequest as e:
            print(e)
            self.cassandra_session = cluster.connect()
            self.cassandra_session.execute("CREATE KEYSPACE active_weather WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 2 }")
            self.cassandra_session = cluster.connect('active_weather')

        # self.__reset__()

    def __reset__(self):
        columns = "id_ int, subject_id text, region int, row int, column int, character text, position int, bitmap list<int>,offset int,height int,width int"

        primary_key = "subject_id,region,row,column,position"
        ordering = "region ASC,row ASC,column ASC, position ASC"

        self.cassandra_session.execute("drop table transcriptions")
        self.cassandra_session.execute("drop table horizontal_lines")
        self.cassandra_session.execute("drop table vertical_lines")
        self.cassandra_session.execute("drop table gold_standard")

        self.cassandra_session.execute("CREATE TABLE transcriptions(" + columns + ", PRIMARY KEY( " + primary_key + ")) WITH CLUSTERING ORDER BY ( " + ordering + ");")
        self.cassandra_session.execute("create index character_index on transcriptions (character)")
        self.cassandra_session.execute("create table gold_standard(subject_id text,region int,row int,column int,cell_contents text, PRIMARY KEY(subject_id,region,row,column))")

        self.cassandra_session.execute("create table horizontal_lines(subject_id text, region int,line_index int,line list<int>,length int, primary key((subject_id,region),line_index)) with clustering order by (line_index ASC)")
        self.cassandra_session.execute("create table vertical_lines(subject_id text, region int,line_index int,line list<int>,length int, primary key((subject_id,region),line_index)) with clustering order by (line_index ASC)")

    def __get_gold_standard__(self,subject_id,region,row,column):
        select_stmt = "select cell_contents from gold_standard where subject_id = '"+ str(subject_id) + "'"
        select_stmt += " and region = " + str(region) + " and row = " + str(row) + " and column = " + str(column)

        results = self.cassandra_session.execute(select_stmt)

        if results == []:
            return None
        else:
            return results[0].cell_contents

    def __get_subjects__(self):
        select_stmt = "select * from gold_standard"
        results = self.cassandra_session.execute(select_stmt)

        print(results)

    def __add_horizontal_lines__(self,subject_id,region,lines):
        for h_index,individual_line in enumerate(lines):
            size = individual_line.shape
            individual_line = individual_line.reshape(size[0]*size[2]).tolist()

            insert_stmt = "insert into horizontal_lines(subject_id,region,line_index,line,length) values("
            insert_stmt += "'"+subject_id+"',"+str(region) + "," + str(h_index) + "," + str(individual_line) + "," + str(size[0]) + ")"
            self.cassandra_session.execute(insert_stmt)

    def __get_horizontal_lines__(self,subject_id,region):
        select_stmt = "select line_index,line,length from horizontal_lines where subject_id ='" + str(subject_id) + "' and region = " + str(region)
        rows = self.cassandra_session.execute(select_stmt)

        all_horizontal_lines = []

        print("loading horizontal lines")
        for index,line,length in rows:
            line = np.asarray(line).reshape(length,2)
            all_horizontal_lines.append(line)

        return all_horizontal_lines

    def __get_vertical_lines__(self,subject_id,region):
        select_stmt = "select line_index,line,length from vertical_lines where subject_id ='" + str(subject_id) + "' and region = " + str(region)
        rows = self.cassandra_session.execute(select_stmt)

        all_vertical_lines = []
        print("loading vertical lines")
        for index,line,length in rows:
            line = np.asarray(line).reshape(length,2)
            all_vertical_lines.append(line)

        return all_vertical_lines

    def __add_vertical_lines__(self,subject_id,region,lines):
        for v_index,individual_line in enumerate(lines):
            size = individual_line.shape
            individual_line = individual_line.reshape(size[0]*size[2]).tolist()

            insert_stmt = "insert into vertical_lines(subject_id,region,line_index,line,length) values("
            insert_stmt += "'"+subject_id+"',"+str(region) + "," + str(v_index) + "," + str(individual_line) + "," + str(size[0]) + ")"
            self.cassandra_session.execute(insert_stmt)

    def __has_cell_been_transcribed__(self,subject_id,region,row,column):
        select_stmt = "select * from gold_standard where subject_id = '"+ str(subject_id) + "'"
        select_stmt += " and region = " + str(region) + " and row = " + str(row) + " and column = " + str(column)

        results = self.cassandra_session.execute(select_stmt)
        # print subject_id,region,row,column

        select_stmt = "select * from gold_standard"
        r = self.cassandra_session.execute(select_stmt)
        return results != []

    def __add_gold_standard__(self,subject_id,region,row,column,cell_contents):
        insert_stmt = "insert into gold_standard(subject_id,region,row,column,cell_contents) values ("
        insert_stmt += "'" + subject_id + "'," + str(region) + "," + str(row) + "," + str(column) + ",'" + str(cell_contents) + "')"
        self.cassandra_session.execute(insert_stmt)

    def __add_character__(self,subject_id,region,row,column,position,character,bitmap,offset):
        assert isinstance(bitmap,np.ndarray)
        height,width = bitmap.shape
        # print subject_id,region,row,column
        bitlist = bitmap.reshape(height*width).tolist()
        insert_stmt = "insert into transcriptions (subject_id,region,row,column,position,character,bitmap,offset,height,width) values ("
        insert_stmt += "'" + subject_id + "'," + str(region) + "," + str(row) + "," + str(column) + "," +str(position)
        insert_stmt += ",'" + character + "'," + str(bitlist) + ","+str(offset)+ "," + str(height) + "," + str(width)+  ")"

        self.cassandra_session.execute(insert_stmt)

    def __get_char__(self,character):
        select_stmt = "select height,width from transcriptions where character = '" + str(character) + "'"

        height_list = []
        width_list = []

        for height,width in self.cassandra_session.execute(select_stmt):
            height_list.append(height)
            width_list.append(width)

        m_height = int(np.median(height_list))
        m_width = int(np.median(width_list))

        select_stmt = "select bitmap,height,width from transcriptions where character = '" + str(character) + "'"

        map_list = []

        stored_bitmap = []

        for bitlist,height,width in self.cassandra_session.execute(select_stmt):
            bitmap = np.asarray(bitlist).reshape((height,width))
            bitmap = bitmap.astype(np.uint8)
            bitmap = cv2.resize(bitmap,(m_height,m_width))

            stored_bitmap.append(bitmap)

            bitmap = bitmap.reshape(m_height*m_width)

            map_list.append(bitmap)

        map_list = np.asarray(map_list)

        pca = PCA(n_components=10)
        X_r = pca.fit(map_list).transform(map_list)

        center = np.mean(X_r,axis=0)

        dist_l = []

        for i,x in enumerate(X_r):
            dist = np.power(np.sum(((x-center)**2)),0.5)
            dist_l.append((i,dist))

        dist_l.sort(key = lambda x:x[1])

        ii = dist_l[-1][0]
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        # fig, ax = plt.subplots()
        im = axes.imshow(stored_bitmap[ii],cmap="gray")
        plt.show()