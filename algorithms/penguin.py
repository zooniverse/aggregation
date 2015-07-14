#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import agglomerative
import aggregation_api

class Penguins(aggregation_api.AggregationAPI):
    def __init__(self):
        aggregation_api.AggregationAPI.__init__(self)

        self.raw_markings,self.raw_classifications = self.__load_classifcations__()
        classification_tasks = {"init":{"shapes":["pt"]}}

        # some stuff to pretend we are a Panoptes project
        classification_tasks = {"init":{"shapes":["pt"]}}
        marking_tasks = {1:1}

        self.workflows = {1:(classification_tasks,marking_tasks)}

        self.cluster_alg = agglomerative.Agglomerative()

    def __load_classifcations__(self):
        # connect to the mongo server
        client = pymongo.MongoClient()
        db = client['penguin_2015-05-08']
        classification_collection = db["penguin_classifications"]

        # clustering = agglomerative.Agglomerative("pt")
        # classifying = classification.VoteCount()

        raw_markings = {"init":{"pt":{}}}
        raw_classifications = {"init":{"pt":{}}}

        task_id = "init"
        shape = "pt"

        animals = []

        users_per_subject = {}

        # sort markings
        for classification in classification_collection.find().limit(500):
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            if zooniverse_id not in raw_markings[task_id][shape]:
                raw_markings[task_id][shape][zooniverse_id] = []
                raw_classifications[task_id][shape][zooniverse_id] = {}

            user_id = classification["user_ip"]

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
                        relevant_params = float(marking["x"]),float(marking["y"])
                        raw_markings[task_id][shape][zooniverse_id].append((user_id,relevant_params,None))

                        # and then the classifications
                        if marking["value"] not in animals:
                            animals.append(marking["value"])
                        animal_index = animals.index(marking["value"])

                        raw_classifications[task_id][shape][zooniverse_id][(relevant_params,user_id)] = animal_index
                except AttributeError:
                    print annotation

        return raw_markings,raw_classifications

    def __sort_markings__(self,workflow_id,subject_set=None,ignore_version=False):
        return self.raw_markings

    def __sort_classifications__(self,workflow_id):
        return self.raw_classifications

project = Penguins()
project.__aggregate__()
# clustering_results = clustering.__aggregate__(raw_markings)
#
# classificaiton_tasks = {"init":{"shapes":["pt"]}}
# print classifying.__aggregate__(raw_classifications,(classificaiton_tasks,{}),clustering_results,users_per_subject)

