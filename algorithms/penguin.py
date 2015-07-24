#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import agglomerative
import aggregation_api
import panoptes_ibcc
import cassandra
import json
from cassandra.concurrent import execute_concurrent
import psycopg2
import urllib2
import os

class Penguins(aggregation_api.AggregationAPI):
    def __init__(self):
        aggregation_api.AggregationAPI.__init__(self)
        self.project_id = -1
        # connect to the mongo server
        client = pymongo.MongoClient()
        db = client['penguin_2015-05-08']
        self.classification_collection = db["penguin_classifications"]
        self.subject_collection = db["penguin_subjects"]

        self.gold_standard = False
        # self.raw_markings,self.raw_classifications = self.__load_classifcations__()

        # some stuff to pretend we are a Panoptes project
        classification_tasks = {1:{"shapes":["point"]}}
        marking_tasks = {1:["point","point","point","point"]}

        database = {}

        self.workflows = {1:(classification_tasks,marking_tasks)}
        self.versions = {1:1}

        self.cluster_algs = {"point":agglomerative.Agglomerative("point")}
        self.classification_alg = panoptes_ibcc.IBCC()

        self.__cassandra_connect__()

        self.classification_table = "penguins_classifications"
        self.users_table = "penguins_users"
        self.subject_id_type = "text"

        self.postgres_session = psycopg2.connect("dbname='zooniverse' user=ggdhines")
        self.postgres_cursor = self.postgres_session.cursor()

        # self.postgres_cursor.execute("create table aggregations (workflow_id int, subject_id text, aggregation jsonb, created_at timestamp, updated_at timestamp)")

    def __get_retired_subjects__(self,workflow_id,with_expert_classifications=None):
        pass

    def __aggregate__(self,workflows=None,subject_set=None):
        # if not gold standard
        if not self.gold_standard:
            aggregation_api.AggregationAPI.__aggregate__(self,workflows,subject_set)

    def __get_aggregated_subjects__(self,workflow_id,num_subjects=20):
        """
        return a list of subjects which have aggregation results
        :param workflow_id:
        :return:
        """
        stmt = "select subject_id from aggregations where workflow_id = " + str(workflow_id)
        self.postgres_cursor.execute(stmt)

        subjects = []

        for r in self.postgres_cursor.fetchall():
            subjects.append(r[0])

        return subjects

    def __image_setup__(self,subject_id,download=True):
        """
        get the local file name for a given subject id and downloads that image if necessary
        :param subject_id:
        :return:
        """
        subject = self.subject_collection.find_one({"zooniverse_id":subject_id})

        url = str(subject["location"]["standard"])
        ii = url.index("www")
        # url = "http://"+url[ii:]

        image_path = aggregation_api.base_directory+"/Databases/images/"+subject_id+".jpg"

        if not(os.path.isfile(image_path)) and download:
            # urllib2.urlretrieve(url, image_path)
            f = open(image_path,"wb")
            f.write(urllib2.urlopen(url).read())
            f.close()

        return image_path

    def __migrate__(self):
        try:
            self.cassandra_session.execute("drop table " + self.classification_table)
            print "table dropped"
        except (cassandra.InvalidRequest,cassandra.protocol.ServerError) as e:
            print "table did not already exist"

        try:
            self.cassandra_session.execute("drop table " + self.users_table)
            print "table dropped"
        except (cassandra.InvalidRequest,cassandra.protocol.ServerError) as e:
            print "table did not already exist"

        self.cassandra_session.execute("CREATE TABLE " + self.classification_table+" (project_id int, workflow_id int, subject_id text, annotations text, user_id text, user_ip inet, workflow_version int, PRIMARY KEY(project_id,workflow_id,workflow_version,subject_id,user_id,user_ip) ) WITH CLUSTERING ORDER BY (workflow_id ASC,workflow_version ASC,subject_id ASC) ;")
        # for looking up which subjects have been classified by specific users
        self.cassandra_session.execute("CREATE TABLE " + self.users_table+ " (project_id int, workflow_id int, workflow_version int, user_id text,user_ip inet,subject_id text, PRIMARY KEY(project_id, workflow_id, workflow_version, user_id,user_ip,subject_id)) WITH CLUSTERING ORDER BY (workflow_id ASC, workflow_version ASC, user_id ASC, user_ip ASC,subject_id ASC);")

        insert_statement = self.cassandra_session.prepare("""
                insert into penguins_classifications (project_id, workflow_id, subject_id,annotations,user_id,user_ip,workflow_version)
                values (?,?,?,?,?,?,?)""")

        user_insert = self.cassandra_session.prepare("""
                insert into penguins_users (project_id, workflow_id, subject_id,user_id,user_ip,workflow_version)
                values (?,?,?,?,?,?)""")

        statements_and_params = []
        statements_and_params2 = []

        all_tools = []

        for ii,classification in enumerate(self.classification_collection.find()):
            if ii % 25000 == 0:
                print ii
                if ii > 0:
                    results = execute_concurrent(self.cassandra_session, statements_and_params)
                    results = execute_concurrent(self.cassandra_session, statements_and_params2)
                    if False in results:
                        print results
                        assert False

                    statements_and_params = []
                    statements_and_params2 = []

            zooniverse_id = classification["subjects"][0]["zooniverse_id"]

            user_name = ""
            if "user_name" in classification:
                user_name = classification["user_name"]
            else:
                user_name = classification["user_ip"]

            user_ip = classification["user_ip"]

            mapped_annotations = [{"task":1,"value":[]}]

            for annotation in classification["annotations"]:

                try:
                    if ("key" not in annotation.keys()) or (annotation["key"] != "marking"):
                        continue
                    for marking in annotation["value"].values():
                        if marking["value"] not in ["adult","chick"]:
                            # print marking["value"]
                            continue

                        # mapped_annotations[index] = marking
                        if marking["value"] not in all_tools:
                            all_tools.append(marking["value"])



                        # mapped_annotations[index]["tool"] = all_tools.index(marking["value"])
                        mapped_annotations[0]["value"].append(marking)
                        mapped_annotations[0]["value"][-1]["tool"] = all_tools.index(marking["value"])

                except (AttributeError,KeyError) as e:
                    # print e
                    pass

            if mapped_annotations == {}:
                continue
            statements_and_params.append((insert_statement,(-1,1,zooniverse_id,json.dumps(mapped_annotations),user_name,user_ip,1)))
            statements_and_params2.append((user_insert,(-1,1,zooniverse_id,user_name,user_ip,1)))

        execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)

    def __store_results__(self,workflow_id,aggregations):
        if self.gold_standard:
            db = "gold_standard_penguins"
        else:
            db = "penguins"

        # try:
        #     self.cassandra_session.execute("drop table " + db)
        # except cassandra.InvalidRequest:
        #     print "table did not already exist"
        #
        # self.cassandra_session.execute("CREATE TABLE " + db + " (zooniverse_id text, aggregations text, primary key(zooniverse_id))")

        insert_statement = self.cassandra_session.prepare("""
                insert into """ + db + """ (zooniverse_id,aggregations)
                values (?,?)""")
        statements_and_params = []
        for zooniverse_id in aggregations:
            statements_and_params.append((insert_statement,(zooniverse_id,json.dumps(aggregations[zooniverse_id]))))

        execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)

    def __get_subject_ids__(self):
        subjects = []
        for subject in self.subject_collection.find({"tutorial":{"$ne":True}}).limit(500):
            subjects.append(subject["zooniverse_id"])

        return subjects



if __name__ == "__main__":
    project = Penguins()
    project.__migrate__()
    # subjects = project.__get_subject_ids__()

    t = 0
    for s in aggregation_api.SubjectGenerator(project):
        t += 1
        project.__aggregate__(workflows=[1],subject_set=s)

        break

        # if t == 15:
        #     break

    # project.__aggregate__(workflows=[1],subject_set=subjects)
    # clustering_results = clustering.__aggregate__(raw_markings)
    #
    # classificaiton_tasks = {"init":{"shapes":["pt"]}}
    # print classifying.__aggregate__(raw_classifications,(classificaiton_tasks,{}),clustering_results,users_per_subject)

