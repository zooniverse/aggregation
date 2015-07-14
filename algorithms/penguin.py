#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import agglomerative
import aggregation_api
import panoptes_ibcc
import cassandra
import json
from cassandra.concurrent import execute_concurrent

class Penguins(aggregation_api.AggregationAPI):
    def __init__(self):
        aggregation_api.AggregationAPI.__init__(self)
        self.project_id = -1
        # connect to the mongo server
        client = pymongo.MongoClient()
        db = client['penguin_2015-06-01']
        self.classification_collection = db["penguin_classifications"]
        self.subject_collection = db["penguin_subjects"]

        self.gold_standard = False
        # self.raw_markings,self.raw_classifications = self.__load_classifcations__()

        # some stuff to pretend we are a Panoptes project
        classification_tasks = {"init":{"shapes":["pt"]}}
        marking_tasks = {1:1}

        self.workflows = {1:(classification_tasks,marking_tasks)}

        self.cluster_alg = agglomerative.Agglomerative()
        self.classification_alg = panoptes_ibcc.IBCC()

        self.__cassandra_connect__()

        self.classification_table = "penguins_classifications"

    def __aggregate__(self,workflows=None,subject_set=None):
        # if not gold standard
        if not self.gold_standard:
            aggregation_api.AggregationAPI.__aggregate__(self,workflows,subject_set)

    def __migrate__(self):
        try:
            self.cassandra_session.execute("drop table " + self.classification_table)
            print "table dropped"
        except cassandra.InvalidRequest:
            print "table did not already exist"

        self.cassandra_session.execute("CREATE TABLE " + self.classification_table+" (project_id int, workflow_id int, subject_id text, annotations text, user_name text, user_ip inet, PRIMARY KEY(project_id,workflow_id,subject_id) ) WITH CLUSTERING ORDER BY (workflow_id ASC,subject_id ASC) ;")

        insert_statement = self.cassandra_session.prepare("""
                insert into penguins_classifications (project_id, workflow_id, subject_id,annotations,user_name,user_ip)
                values (?,?,?,?,?,?)""")

        statements_and_params = []

        all_tools = []

        for ii,classification in enumerate(self.classification_collection.find().limit(500000)):
            if ii % 25000 == 0:
                print ii
                if ii > 0:
                    execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)
                    statements_and_params = []

            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            user_ip = classification["user_ip"]

            user_name = ""
            if "user_name" in classification:
                user_name = classification["user_name"]

            user_ip = classification["user_ip"]

            mapped_annotations = {}

            for annotation in classification["annotations"]:
                try:
                    if ("key" not in annotation.keys()) or (annotation["key"] != "marking"):
                        continue
                    for index,marking in enumerate(annotation["value"].values()):
                        mapped_annotations[index] = marking
                        if marking["value"] not in all_tools:
                            all_tools.append(marking["value"])

                        mapped_annotations[index]["tool"] = all_tools.index(marking["value"])

                except AttributeError:
                    pass

            if mapped_annotations == {}:
                continue
            statements_and_params.append((insert_statement,(-1,1,zooniverse_id,json.dumps(mapped_annotations),user_name,user_ip)))

        execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)

    def __load_subjects__(self,workflow_id):
        subject_set = []
        for subject in self.subject_collection.find().limit(500000):
            subject_set.append(subject["zooniverse_id"])

        return subject_set

    def __load_classifcations__(self):
        experts = ["caitlin.black"]



        raw_markings = {"init":{"pt":{}}}
        raw_classifications = {"init":{"pt":{}}}

        task_id = "init"
        shape = "pt"

        animals = []

        users_per_subject = {}

        # sort markings
        for ii,classification in enumerate(classification_collection.find().limit(5000000)):
            if ii % 25000 == 0:
                print ii
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            if zooniverse_id not in raw_markings[task_id][shape]:
                raw_markings[task_id][shape][zooniverse_id] = []
                raw_classifications[task_id][shape][zooniverse_id] = {}

            user_id = classification["user_ip"]

            if "user_name" in classification:
                user_name = classification["user_name"]
                # todo - clean this up
                if self.gold_standard :
                    if user_name not in experts:
                        continue
                else:
                    if user_name in experts:
                        continue
            # if the user was not logged in, we assume that it could not be the expert
            elif self.gold_standard:
                continue

            if zooniverse_id not in users_per_subject:
                users_per_subject[zooniverse_id] = []

            users_per_subject[zooniverse_id].append(user_id)

            for annotation in classification["annotations"]:
                assert isinstance(annotation,dict)
                # print annotation.keys()
                if ("key" not in annotation.keys()) or (annotation["key"] != "marking"):
                    continue
                try:
                    for marking in annotation["value"].values():
                        # deal with markings first
                        try:
                            relevant_params = float(marking["x"]),float(marking["y"])
                        except ValueError:
                            continue

                        assert isinstance(raw_markings[task_id][shape][zooniverse_id],list)
                        raw_markings[task_id][shape][zooniverse_id].append((user_id,relevant_params,None))

                        # and then the classifications
                        if marking["value"] not in animals:
                            animals.append(marking["value"])
                        animal_index = animals.index(marking["value"])

                        raw_classifications[task_id][shape][zooniverse_id][(relevant_params,user_id)] = animal_index
                except AttributeError:
                    pass


        return raw_markings,raw_classifications

    # def __sort_markings__(self,workflow_id,subject_set=None,ignore_version=False):
    #     return self.raw_markings
    #
    # def __sort_classifications__(self,workflow_id):
    #     return self.raw_classifications

    def __store_results__(self,workflow_id,aggregations):
        if self.gold_standard:
            db = "gold_standard_penguins"
        else:
            db = "penguins"

        try:
            self.cassandra_session.execute("drop table " + db)
        except cassandra.InvalidRequest:
            print "table did not already exist"

        self.cassandra_session.execute("CREATE TABLE " + db + " (zooniverse_id text, aggregations text, primary key(zooniverse_id))")

        insert_statement = self.cassandra_session.prepare("""
                insert into """ + db + """ (zooniverse_id,aggregations)
                values (?,?)""")
        statements_and_params = []
        for zooniverse_id in aggregations:
            statements_and_params.append((insert_statement,(zooniverse_id,json.dumps(aggregations[zooniverse_id]))))

        execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)

project = Penguins()
project.__migrate__()
project.__aggregate__()
# clustering_results = clustering.__aggregate__(raw_markings)
#
# classificaiton_tasks = {"init":{"shapes":["pt"]}}
# print classifying.__aggregate__(raw_classifications,(classificaiton_tasks,{}),clustering_results,users_per_subject)

