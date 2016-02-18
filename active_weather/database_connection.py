from cassandra.cluster import Cluster,InvalidRequest
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

        columns = "id_ int, subject_id text, region int, row int, column int, character text, position int, bitmap list<int>,offset int,height int,width int"

        primary_key = "subject_id,region,row,column,position"
        ordering = "region ASC,row ASC,column ASC, position ASC"

        # self.cassandra_session.execute("drop table transcriptions")
        # self.cassandra_session.execute("CREATE TABLE transcriptions(" + columns + ", PRIMARY KEY( " + primary_key + ")) WITH CLUSTERING ORDER BY ( " + ordering + ");")
        # self.cassandra_session.execute("create index character_index on transcriptions (character)")

    def __has_cell_been_transcribed__(self,subject_id,region,row,column):
        select_stmt = "select * from transcriptions where subject_id = '"+ str(subject_id) + "'"
        select_stmt += " and region = " + str(region) + " and row = " + str(row) + " and column = " + str(column)

        results = self.cassandra_session.execute(select_stmt)
        # print subject_id,region,row,column
        # print results
        return results != []

    def __add_transcription__(self,subject_id,region,row,column,position,character,bitmap,offset):
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

        print len(map_list)

        map_list = np.asarray(map_list)

        print map_list.shape

        pca = PCA(n_components=10)
        X_r = pca.fit(map_list).transform(map_list)

        print sum(pca.explained_variance_ratio_)

        center = np.mean(X_r,axis=0)

        dist_l = []

        for i,x in enumerate(X_r):
            dist = np.power(np.sum(((x-center)**2)),0.5)
            dist_l.append((i,dist))

        dist_l.sort(key = lambda x:x[1])
        print dist_l

        ii = dist_l[10][0]
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        # fig, ax = plt.subplots()
        im = axes.imshow(stored_bitmap[ii],cmap="gray")
        plt.show()