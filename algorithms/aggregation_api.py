#!/usr/bin/env python

import os
import yaml
import urllib2
import psycopg2
import cookielib
import re
import json
import cassandra
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent
import urllib
import datetime
import classification
# import clustering_dict
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
from matplotlib.patches import Ellipse
# from clustering import  cnames
# import numpy
# import datetime
import sys
from PIL import Image
import agglomerative
import cluster_count
import clustering
import blob_clustering


if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

class InvalidMarking(Exception):
    def __init__(self,pt):
        self.pt = pt
    def __str__(self):
        return "invalid marking: " + str(self.pt)


class ImageNotDownloaded(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'image not downloaded'


class ImproperTool(Exception):
    def __init__(self,tool):
        self.tool = tool
    def __str__(self):
        return "improper tool: " + str(self.tool)


def line_mapping(marking,image_dimensions):
    # want to extract the params x1,x2,y1,y2 but
    # ALSO make sure that x1 <= x2 and flip if necessary
    x1 = marking["x1"]
    x2 = marking["x2"]
    y1 = marking["y1"]
    y2 = marking["y2"]

    if min(x1,x2,y1,y2) < 0:
        raise InvalidMarking(marking)

    if (max(x1,x2) >= image_dimensions[0]) or (max(y1,y2) >= image_dimensions[1]):
        raise InvalidMarking(marking)

    if x1 <= x2:
        return x1,x2,y1,y2
    else:
        return x2,x1,y2,y1


def point_mapping(marking,image_dimensions):
    # todo - this has to be changed
    image_dimensions = 1000,1000
    x = float(marking["x"])
    y = float(marking["y"])

    if (x<0)or(y<0)or(x > image_dimensions[0]) or(y>image_dimensions[1]):
        raise InvalidMarking(marking)

    return x,y


def rectangle_mapping(marking,image_dimensions):
    x = marking["x"]
    y = marking["y"]

    x2 = x + marking["width"]
    y2 = y + marking["height"]

    if (x<0)or(y<0)or(x2 > image_dimensions[0]) or(y2>image_dimensions[1]):
        raise InvalidMarking(marking)

    # return x,y,x2,y2
    return (x,y),(x,y2),(x2,y2),(x2,y)


def ellipse_mapping(marking,image_dimensions):
    return marking["x"],marking["y"],marking["rx"],marking["ry"],marking["angle"]


class AggregationAPI:
    def __init__(self,project=None):#,user_threshold= None, score_threshold= None): #Supernovae
        self.cluster_algs = None
        self.classification_alg = None
        self.workflows = None
        self.versions = None
        self.classifications = None

        self.marking_params_per_shape = dict()
        self.marking_params_per_shape["line"] = line_mapping
        self.marking_params_per_shape["point"] = point_mapping
        self.marking_params_per_shape["ellipse"] = ellipse_mapping
        self.marking_params_per_shape["rectangle"] = rectangle_mapping

        # only continue the set up if the project name is given
        self.project_short_name = project

        if project is None:
            return

        self.classification_table = "classifications"

        # get my userID and password
        # purely for testing, if this file does not exist, try opening on Greg's computer
        try:
            panoptes_file = open("config/aggregation.yml","rb")
        except IOError:
            panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")
        api_details = yaml.load(panoptes_file)

        # details for connecting to Panoptes
        self.user_name = api_details[project]["name"]
        self.password = api_details[project]["password"]
        self.host = api_details[project]["host"] #"https://panoptes-staging.zooniverse.org/"
        self.host_api = self.host+"api/"
        self.owner = api_details[project]["owner"] #"brian-testing" or zooniverse
        self.project_name = api_details[project]["project_name"]
        self.app_client_id = api_details[project]["app_client_id"]
        self.environment = api_details[project]["environment"]
        self.token = None

        # the http api for connecting to Panoptes
        self.http_api = None
        print "connecting to Panoptes http api"
        # set the http_api and basic project details
        self.__panoptes_connect__()

        # # details about the project we are going to work with
        try:
            self.project_id = api_details[project]["project_id"]
        except KeyError:
            self.project_id = self.__get_project_id()

        # # get the most recent workflow version
        # # todo - probably need to make sure that we only read in classifications from the most recent workflow version
        # try:
        #     self.workflow_version = api_details[project]["workflow_version"]
        # except KeyError:
        #     self.workflow_version = self.__get_workflow_version()
        #
        # try:
        #     self.workflow_id = api_details[project]["workflow_id"]
        # except KeyError:
        #     self.workflow_id = self.__get_workflow_id()
        # print self.workflow_id
        # assert False
        # assert self.workflow_id is not None



        # now connect to the Panoptes db - postgres
        try:
            database_file = open("config/database.yml")
        except IOError:
            database_file = open(base_directory+"/Databases/database.yml")

        database_details = yaml.load(database_file)
        self.postgres_session = None

        self.__postgres_connect(database_details)

        # and to the cassandra db as well
        self.__cassandra_connect__()

        # there may be more than one workflow associated with a project - read them all in
        # and set up the associated tasks
        self.workflows = self.__setup_workflows__()
        self.versions = self.__get_workflow_versions__()

        # can have different clustering algorithms for different shapes

        #
        #
        # self.workflows = {}
        # self.__load_workflows__()
        # # if there is only one workflow for the project, just use that
        # # otherwise, we will need to specify which workflow we want to use
        # # self.workflow_id = None
        # # subjects don't have to be the same for all workflows for the same project - so only load the subjects
        # # if there workflow is given (e.g. there is only one workflow)
        # self.subjects = []
        # if len(self.workflows) == 1:
        #     self.workflow_id = self.workflows.keys()[0]
        #

        # load in the subjects for the project
        # self.__get_subject_ids__()

        #     # load the tasks associated with this workflow
        #     self.__task_setup__()
        #

        # self.aggregations = {}
        #

    def __aggregate__(self,workflows=None,subject_set=None):
        if workflows is None:
            workflows = self.workflows

        for workflow_id in workflows:
            if subject_set is None:
                # subject_set = self.__get_retired_subjects__(workflow_id)
                subject_set = self.__load_subjects__(workflow_id)
            print "aggregating " + str(len(subject_set)) + " subjects"
            # self.__describe__(workflow_id)
            classification_tasks,marking_tasks = self.workflows[workflow_id]

            clustering_aggregations = None
            classification_aggregations = None

            # if we have provided a clustering algorithm and there are marking tasks
            # ideally if there are no marking tasks, then we shouldn't have provided a clustering algorithm
            # but nice sanity check

            if (self.cluster_algs is not None) and (marking_tasks != {}):
                print "clustering"
                clustering_aggregations = self.__cluster__(workflow_id,subject_set)
                # assert (clustering_aggregations != {}) and (clustering_aggregations is not None)
            if (self.classification_alg is not None) and (classification_tasks != {}):
                # we may need the clustering results
                print "classifying"
                classification_aggregations = self.__classify__(workflow_id,clustering_aggregations,subject_set)

            # if we have both markings and classifications - we need to merge the results
            if (clustering_aggregations is not None) and (classification_aggregations is not None):
                aggregations = self.__merge_aggregations__(clustering_aggregations,classification_aggregations)
            elif clustering_aggregations is None:
                aggregations = classification_aggregations
            else:
                aggregations = clustering_aggregations

            # finally, store the results
            self.__upsert_results__(workflow_id,aggregations)

    def __cassandra_connect__(self):
        """
        connect to the AWS instance of Cassandra - try 10 times and raise an error
        :return:
        """
        for i in range(10):
            try:
                self.cluster = Cluster()#['panoptes-cassandra.zooniverse.org'],protocol_version = 3)
                self.cassandra_session = self.cluster.connect('zooniverse')
                return
            except cassandra.cluster.NoHostAvailable:
                pass

        assert False

    def __chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

    def __classify__(self,workflow_id,clustering_aggregations,subject_set=None):
        # get the raw classifications for the given workflow
        raw_classifications = self.__sort_classifications__(workflow_id,subject_set)
        if raw_classifications == {}:
            print "returning empty"
            empty_aggregation = {"param":"subject_id"}
            # for subject_set in empty_aggregation
            return empty_aggregation

        return self.classification_alg.__aggregate__(raw_classifications,self.workflows[workflow_id],clustering_aggregations)

    def __cluster__(self,workflow_id,subject_set=None):
        """
        run the clustering algorithm for a given workflow
        need to have already checked that the workflow requires clustering
        :param workflow_id:
        :return:
        """
        # assert (self.cluster_algs != {}) and (self.cluster_algs is not None)
        print "workflow id is " + str(workflow_id)
        # get the raw classifications for the given workflow
        if subject_set is None:
            subject_set = self.__load_subjects__(workflow_id)

        raw_markings = self.__sort_markings__(workflow_id,subject_set)

        if raw_markings == {}:
            print "warning - empty set of images"
            print subject_set
            return {}
        # assert raw_markings != {}
        # assert False

        # will store the aggregations for all clustering
        cluster_aggregation = {}
        for shape in self.cluster_algs:
            print "shape is " + shape
            shape_aggregation = self.cluster_algs[shape].__aggregate__(raw_markings)

            if cluster_aggregation == {}:
                cluster_aggregation = shape_aggregation
            else:
                assert isinstance(cluster_aggregation,dict)
                cluster_aggregation = self.__merge_aggregations__(cluster_aggregation,shape_aggregation)

                # print json.dumps(cluster_aggregation, sort_keys=True, indent=4, separators=(',', ': '))
                # assert False

        fnames = {}
        # for s in subjects[0:20]:
        #     try:
        #         print s
        #         fnames[s] = self.__image_setup__(s,download=False)
        #     except ImageNotDownloaded:
        #         break

        # go through each shape separately and merge the results in
        # for shape in self.cluster_algs:
        #     print shape
        #     shape_aggregation = self.cluster_algs[shape].__aggregate__(raw_markings,fnames)
        #
        #     # only merge if we have some results
        #     if shape_aggregation != {"param":"subject_id"}:
        #         cluster_aggregation = self.__merge_aggregations__(cluster_aggregation,shape_aggregation)

        return cluster_aggregation


    # def __describe__(self,workflow_id):
    #     select = "SELECT tasks from workflows where id = " + str(workflow_id)
    #     self.postgres_cursor.execute(select)
    #     tasks = self.postgres_cursor.fetchone()[0]
    #
    #     select = "SELECT strings from workflow_contents where id = " + str(workflow_id)
    #     self.postgres_cursor.execute(select)
    #     contents = self.postgres_cursor.fetchone()[0]
    #
    #     self.description = {}
    #     print "===---"
    #     for task_id in tasks:
    #         print tasks[task_id].keys()
    #         # print task_id
    #         self.description[task_id] = []
    #         # print tasks[task_id]
    #         if "question" in tasks[task_id]:
    #             question = tasks[task_id]["question"]
    #             self.description[task_id].append(contents[question])
    #             # print contents[question]
    #             answers = tasks[task_id]["answers"]
    #             # print answers
    #             for ans in answers:
    #                 # print ans
    #                 label = ans["label"]
    #                 labels = label.split(".")
    #                 # question_index = labels[2]
    #                 self.description[task_id].append(contents[label])
    #                 # print self.description[task_id][-1]
    #         else:
    #             assert "tools" in tasks[task_id]
    #             print tasks[task_id]["tools"]
    #     # self. description

    def __get_num_clusters__(self,subject_id,task_id):
        return len(self.cluster_alg.clusterResults[subject_id][task_id])

    def __get_classifications__(self,subject_id,task_id,cluster_index=None,question_id=None):
        # either both of these variables are None or neither of them are
        assert (cluster_index is None) == (question_id is None)

        if cluster_index is None:
            return self.classifications[subject_id][task_id]
        else:
            return self.classifications[subject_id][task_id][cluster_index][question_id]

    def __get_project_id(self):
        """
        get the id number for our project
        this is mostly a "beta" feature so that given a name of a project, I can find the id
        in practice, aggregation will be run with just giving the project id
        :return:
        """
        # print self.host_api+"projects?owner="+urllib2.quote(self.owner)+"&display_name="+urllib2.quote(self.project_name)
        # assert False
        # request = urllib2.Request(self.host_api+"projects?owner="+urllib2.quote(self.owner)+"&display_name="+urllib2.quote(self.project_name))
        # print self.host_api+"projects?display_name="+urllib2.quote(self.project_name)
        request = urllib2.Request(self.host_api+"projects?display_name="+urllib2.quote(self.project_name))
        # request = urllib2.Request(self.host_api+"projects?owner="+self.owner+"&display_name=Galaxy%20Zoo%20Bar%20Lengths")
        # print hostapi+"projects?owner="+owner+"&display_name="+project_name
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)

        # request
        try:
            response = urllib2.urlopen(request)
            body = response.read()
        except urllib2.HTTPError as e:
            print self.host_api+"projects?owner="+self.owner+"&display_name="+self.project_name
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
            print 'Error response body: ', e.read()
            raise
        except urllib2.URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
            raise

        data = json.loads(body)
        # put it in json structure and extract id
        return data["projects"][0]["id"]
        # return None

    def __get_retired_subjects__(self,workflow_id):
        retired_subjects = []

        stmt = """SELECT * FROM "subjects"
            INNER JOIN "set_member_subjects" ON "set_member_subjects"."subject_id" = "subjects"."id"
            INNER JOIN "subject_sets" ON "subject_sets"."id" = "set_member_subjects"."subject_set_id"
            INNER JOIN "subject_sets_workflows" ON "subject_sets_workflows"."subject_set_id" = "subject_sets"."id"
            INNER JOIN "workflows" ON "workflows"."id" = "subject_sets_workflows"."workflow_id"
            WHERE "workflows"."id" = """+str(workflow_id)+ """ AND (""" + str(workflow_id) + """ = ANY("set_member_subjects"."retired_workflow_ids"))"""

        self.postgres_cursor.execute(stmt)
        for subject in self.postgres_cursor.fetchall():
            retired_subjects.append(subject[0])

        return retired_subjects

    def __get_results__(self,workflow_id,subject_id=None):
        stmt = "select * from aggregations where workflow_id = " + str(workflow_id)
        if subject_id is not None:
            stmt += " and subject_id = " + str(subject_id)
        self.postgres_cursor.execute(stmt)
        all_results = []
        for r in self.postgres_cursor.fetchall():
            # print json.dumps(r[3], sort_keys=True,indent=4, separators=(',', ': '))
            # print r[3].keys()
            assert isinstance(r[3],dict)
            results = {r[2]:r[3]}
            all_results.append(results)
        with open('/home/greg/data.txt', 'w') as outfile:
            json.dump(all_results, outfile,sort_keys=True,indent=4, separators=(',', ': '))


    def __get_workflow_details__(self,workflow_id):
        request = urllib2.Request(self.host_api+"workflows?project_id="+str(self.project_id))
        # request = urllib2.Request(self.host_api+"workflows/project_id="+str(self.project_id))
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)

        # request
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError as e:
            sys.stderr.write('The server couldn\'t fulfill the request.\n')
            sys.stderr.write('Error code: ' + str(e.code) + "\n")
            sys.stderr.write('Error response body: ' + str(e.read()) + "\n")
            raise
        except urllib2.URLError as e:
            sys.stderr.write('We failed to reach a server.\n')
            sys.stderr.write('Reason: ' + str(e.reason) + "\n")
            raise
        else:
            # everything is fine
            body = response.read()

        # put it in json structure and extract id
        data = json.loads(body)

        instructions = {}
        for workflows in data["workflows"]:
            if int(workflows["id"]) == workflow_id:

                for task_id,task in workflows["tasks"].items():
                    # print task
                    # print task["tools"]["details"]
                    # print

                    instructions[task_id] = {}
                    # classification task
                    if "question" in task:
                        instructions[task_id]["instruction"] = task["question"]
                        instructions[task_id]["answers"] = {}
                        for answer_id,answer in enumerate(task["answers"]):
                            label = answer["label"]
                            label = re.sub("'","",label)
                            instructions[task_id]["answers"][answer_id] = label

                    else:
                        instructions[task_id]["instruction"] = task["instruction"]

                        instructions[task_id]["tools"] = {}
                        for tool_index,tool in enumerate(task["tools"]):
                            instructions[task_id]["tools"][tool_index] = {}
                            instructions[task_id]["tools"][tool_index]["question"] = tool["label"]
                            if tool["details"] != []:
                                instructions[task_id]["tools"][tool_index]["followup_questions"] = {}

                                for subtask_index,subtask in enumerate(tool["details"]):
                                    instructions[task_id]["tools"][tool_index]["followup_questions"][subtask_index] = {}
                                    instructions[task_id]["tools"][tool_index]["followup_questions"][subtask_index]["question"] = subtask["question"]
                                    instructions[task_id]["tools"][tool_index]["followup_questions"][subtask_index]["answers"] = {}
                                    #print subtask["question"]
                                    for answer_index,answers in enumerate(subtask["answers"]):
                                        instructions[task_id]["tools"][tool_index]["followup_questions"][subtask_index]["answers"][answer_index] = answers

                        # print tool["label"]
                    # print task

                return instructions

        assert False


    def __get_workflow_versions__(self):#,project_id):
        request = urllib2.Request(self.host_api+"workflows?project_id="+str(self.project_id))
        # request = urllib2.Request(self.host_api+"workflows/project_id="+str(self.project_id))
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)

        # request
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError as e:
            sys.stderr.write('The server couldn\'t fulfill the request.\n')
            sys.stderr.write('Error code: ' + str(e.code) + "\n")
            sys.stderr.write('Error response body: ' + str(e.read()) + "\n")
            raise
        except urllib2.URLError as e:
            sys.stderr.write('We failed to reach a server.\n')
            sys.stderr.write('Reason: ' + str(e.reason) + "\n")
            raise
        else:
            # everything is fine
            body = response.read()

        # put it in json structure and extract id
        data = json.loads(body)

        versions = {}

        for workflows in data["workflows"]:
            # if int(workflows["id"]) == workflow_id:
            versions[int(workflows["id"])] = int(math.floor(float(workflows["version"])))
            # versions[int(w["id"])] = w["version"] #int(math.floor(float(w["version"])))

        return versions

    def __image_setup__(self,subject_id,download=True):
        """
        get the local file name for a given subject id and downloads that image if necessary
        :param subject_id:
        :return:
        """
        request = urllib2.Request(self.host_api+"subjects/"+str(subject_id))
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)
        # request
        try:
            response = urllib2.urlopen(request)
            body = response.read()
        except urllib2.HTTPError as e:
            print self.host_api+"subjects/"+str(subject_id)
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
            print 'Error response body: ', e.read()
            raise
        except urllib2.URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
            raise

        data = json.loads(body)

        # print data["subjects"]
        # assert False

        url = str(data["subjects"][0]["locations"][0]["image/jpeg"])

        slash_index = url.rfind("/")
        fname = url[slash_index+1:]

        image_path = base_directory+"/Databases/images/"+fname

        if not(os.path.isfile(image_path)):
            if download:
                print "downloading"
                urllib.urlretrieve(url, image_path)
            raise ImageNotDownloaded()

        return image_path

    # def __list_all_versions__(self):
    #     request = urllib2.Request(self.host_api+"workflows/6/versions?")
    #     request.add_header("Accept","application/vnd.api+json; version=1")
    #     request.add_header("Authorization","Bearer "+self.token)
    #
    #     # request
    #     try:
    #         response = urllib2.urlopen(request)
    #     except urllib2.HTTPError as e:
    #         print 'The server couldn\'t fulfill the request.'
    #         print 'Error code: ', e.code
    #         print 'Error response body: ', e.read()
    #         raise
    #     except urllib2.URLError as e:
    #         print 'We failed to reach a server.'
    #         print 'Reason: ', e.reason
    #         raise
    #     else:
    #         # everything is fine
    #         body = response.read()
    #
    #     # put it in json structure and extract id
    #     data = json.loads(body)
    #     print data["meta"].keys()
    #     print data["versions"]

    def __load_subjects__(self,workflow_id):
        """
        load the list of subject ids from Cassandra
        :param workflow_id:
        :return:
        """
        # version = int(math.floor(float(self.versions[workflow_id])))
        stmt = "SELECT subject_id FROM subjects WHERE project_id = " + str(self.project_id) + " and workflow_id = " + str(workflow_id)# + " and workflow_version = " + str(version)
        subjects = set([r.subject_id for r in self.cassandra_session.execute(stmt)])

        assert subjects != ()

        return list(subjects)

    def __merge_aggregations__(self,agg1,agg2):
        """
        merge aggregations - could be clustering and classification aggregations
        could be two different clustering aggregations for different shapes
        :param clust_agg:
        :param class_agg:
        :return:
        """
        # start with the clustering results and merge in the classification results
        # assert (agg1 is not None) or (agg2 is not None)
        assert isinstance(agg1,dict) and isinstance(agg2,dict)

        if agg1 == {}:
            return agg2
        elif agg2 == {}:
            return agg1

        for kw in agg2:
            if kw not in agg1:
                agg1[kw] = agg2[kw]
            elif agg1[kw] != agg2[kw]:
                try:
                    agg1[kw] = self.__merge_aggregations__(agg1[kw],agg2[kw])
                except TypeError:
                    print "====-----"
                    print type(agg1)
                    print type(agg2)
                    print agg1
                    print agg2
                    print kw
                    assert False
        return agg1

    def __migrate__(self):
        # tt = set([465026, 465003, 493062, 492809, 465034, 465172, 493205, 465048, 465177, 493211, 464965, 492960, 465057, 465058, 492707, 492836, 465121, 492975, 464951, 464952, 464953, 464954, 464955, 464956, 464957, 464958, 464959, 464960, 464961, 492611, 492741, 492615, 465100, 492623, 492728, 492626, 492886, 464975, 464988, 492897, 464998, 492776, 492907, 492914, 465019, 492669])
        # print self.versions
        # assert False
        # try:
        #     self.cassandra_session.execute("drop table classifications")
        #     self.cassandra_session.execute("drop table subjects")
        #     print "table dropped"
        # except cassandra.InvalidRequest:
        #     print "table did not already exist"
        #
        # self.cassandra_session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, created_at timestamp,annotations text,  updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, subject_id int, workflow_version int,metadata text, PRIMARY KEY(project_id,workflow_id,subject_id,workflow_version,user_ip,user_id) ) WITH CLUSTERING ORDER BY (workflow_id ASC,subject_id ASC,workflow_version ASC,user_ip ASC,user_id ASC);")
        # self.cassandra_session.execute("CREATE TABLE subjects (project_id int, workflow_id int, workflow_version int, subject_id int, PRIMARY KEY(project_id,workflow_id,subject_id,workflow_version));")
        # except
        #     print "table did not exist"
        #     pass

        # try:
        #             # except cassandra.AlreadyExists:
        #     pass

        subject_listing = set()


        select = "SELECT * from classifications where project_id="+str(self.project_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)

        self.migrated_subjects = set()
        print "trying to migrate " + str(self.project_id)
        insert_statement = self.cassandra_session.prepare("""
                insert into classifications (project_id, user_id, workflow_id,  created_at,annotations, updated_at, user_group_id, user_ip, completed, gold_standard, subject_id, workflow_version,metadata)
                values (?,?,?,?,?,?,?,?,?,?,?,?,?)""")



        statements_and_params = []
        migrated = {}
        for ii,t in enumerate(cur.fetchall()):
            print ii
            id_,project_id,user_id,workflow_id,annotations,created_at,updated_at,user_group_id,user_ip,completed,gold_standard,expert_classifier,metadata,subject_ids,workflow_version = t
            # print t
            # assert False
            assert len(subject_ids) == 1
            self.migrated_subjects.add(subject_ids[0])

            if gold_standard != True:
                gold_standard = False

            if not isinstance(user_group_id,int):
                user_group_id = -1

            if not isinstance(user_id,int):
                user_id = -1
            # get only the major version of the workflow
            workflow_version = int(math.floor(float(workflow_version)))
            id = workflow_id,subject_ids[0]
            # if subject_ids[0] == 4153:
            #     print workflow_id,user_ip

            if id not in migrated:
                migrated[id] = 0
            migrated[id] += 1

            params = (project_id, user_id, workflow_id,created_at, json.dumps(annotations), updated_at, user_group_id, user_ip,  completed, gold_standard,  subject_ids[0], workflow_version,json.dumps(metadata))
            statements_and_params.append((insert_statement, params))

            # params2 = (project_id,workflow_id,workflow_version,subject_ids[0])
            # statements_and_params.append((insert_statement2,params2))
            subject_listing.add((project_id,workflow_id,workflow_version,subject_ids[0]))

            if len(statements_and_params) == 100:
                results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)
                statements_and_params = []
                # print results

        # insert any "left over" classifications
        if statements_and_params != []:
            results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)
        # stmt = "select count(*) from subjects"
        # print migrated
        # print self.cassandra_session.execute(stmt)

        # now update the subject ids
        statements_and_params = []
        insert_statement = self.cassandra_session.prepare("""
                insert into subjects (project_id,workflow_id,workflow_version,subject_id)
                values (?,?,?,?)""")
        for s in subject_listing:
            statements_and_params.append((insert_statement, s))

            if len(statements_and_params) == 100:
                results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)
                statements_and_params = []
        if statements_and_params != []:
            results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)

    def __panoptes_connect__(self):
        """
        make the main connection to Panoptes - through http
        :return:
        """
        for i in range(20):
            try:
                print "connecting to Pantopes, attempt: " + str(i)
                cj = cookielib.CookieJar()
                opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

                #1. get the csrf token

                request = urllib2.Request(self.host+"users/sign_in",None)
                response = opener.open(request)

                body = response.read()
                # grep for csrf-token
                try:
                    csrf_token = re.findall(".+csrf-token.\s+content=\"(.+)\"",body)[0]
                except IndexError:
                    print body
                    raise

                #2. use the token to get a devise session via JSON stored in a cookie
                devise_login_data=("{\"user\": {\"login\":\""+self.user_name+"\",\"password\":\""+self.password+
                                   "\"}, \"authenticity_token\": \""+csrf_token+"\"}")

                request = urllib2.Request(self.host+"users/sign_in",data=devise_login_data)
                request.add_header("Content-Type","application/json")
                request.add_header("Accept","application/json")

                try:
                    response = opener.open(request)
                except urllib2.HTTPError as e:
                    print 'In get_bearer_token, stage 2:'
                    print 'The server couldn\'t fulfill the request.'
                    print 'Error code: ', e.code
                    print 'Error response body: ', e.read()
                    raise
                except urllib2.URLError as e:
                    print 'We failed to reach a server.'
                    print 'Reason: ', e.reason
                    raise
                else:
                    # everything is fine
                    body = response.read()

                #3. use the devise session to get a bearer token for API access
                if self.app_client_id != "":
                    bearer_req_data=("{\"grant_type\":\"password\",\"client_id\":\"" + self.app_client_id + "\"}")
                else:
                    bearer_req_data=("{\"grant_type\":\"password\"}")
                request = urllib2.Request(self.host+"oauth/token",bearer_req_data)
                request.add_header("Content-Type","application/json")
                request.add_header("Accept","application/json")

                try:
                    response = opener.open(request)
                except urllib2.HTTPError as e:
                    print 'In get_bearer_token, stage 3:'
                    print 'The server couldn\'t fulfill the request.'
                    print 'Error code: ', e.code
                    print 'Error response body: ', e.read()
                    raise
                except urllib2.URLError as e:
                    print 'We failed to reach a server.'
                    print 'Reason: ', e.reason
                    raise
                else:
                    # everything is fine
                    body = response.read()

                # extract the bearer token
                json_data = json.loads(body)
                bearer_token = json_data["access_token"]

                self.token = bearer_token
                break
            except (urllib2.HTTPError,urllib2.URLError) as e:
                print "trying to connect/init again again"
                pass



    def __plot_image__(self,subject_id):
        fname = self.__image_setup__(subject_id)

        for i in range(10):
            try:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                image_file = cbook.get_sample_data(fname)
                image = plt.imread(image_file)
                # fig, ax = plt.subplots()
                im = ax.imshow(image)

                return
            except IOError:
                # try downloading that image again
                os.remove(fname)
                self.__image_setup__(subject_id)

    def __plot_individual_points__(self,subject_id,task_id,shape):
        for cluster in self.cluster_alg.clusterResults[task_id][shape][subject_id]:
            for pt in cluster["points"]:
                if shape == "line":
                    plt.plot([pt[0],pt[1]],[pt[2],pt[3]],color="red")
                elif shape == "point":
                    plt.plot([pt[0]],[pt[1]],".",color="red")
                elif shape == "circle":
                    print (pt[0],pt[1]),pt[2]
                    e = Ellipse((pt[0],pt[1]),width = pt[2],height=pt[2],fill=False,color="red")
                    # circle = plt.Circle((pt[0],pt[1]),pt[2],color=cnames.values()[users.index(user_id)])
                    plt.gca().add_patch(e)
                    # ax.add_artist(e)
                    # e.set_alpha(0)
                elif shape == "ellipse":
                    # ("angle","rx","ry","x","y")
                    e = Ellipse((pt[3],pt[4]),width = pt[2],height=pt[1],fill=False,angle=pt[0],color="red")
                elif shape == "rectangle":
                    plt.plot([pt[0],pt[0]+pt[2]],[pt[1],pt[1]],color="red")
                    plt.plot([pt[0],pt[0]],[pt[1],pt[1]+pt[3]],color="red")
                    plt.plot([pt[0]+pt[2],pt[0]+pt[2]],[pt[1],pt[1]+pt[3]],color="red")
                    plt.plot([pt[0],pt[0]+pt[2]],[pt[1]+pt[3],pt[1]+pt[3]],color="red")
                else:
                    print shape
                    assert False

        plt.axis('scaled')

    def __plot_cluster_results__(self,subject_id,task_id,shape):
        # for task in self.cluster_alg.clusterResults[subject_id]:
        #     not really a task - just there to make things easier to understand
            # if task == "param":
            #     continue
        for cluster in self.cluster_alg.clusterResults[task_id][shape][subject_id]:
            center = cluster["center"]
            points = cluster["points"]
            if shape == "line":
                plt.plot([center[0],center[1]],[center[2],center[3]],color="blue")
            elif shape == "point":
                for pt in points:
                    plt.plot([pt[0],],[pt[1],],'.',color="red")
                plt.plot([center[0],],[center[1],],"o",color="blue")
            else:
                assert False
            # plt.title("number of users: " + str(len(cluster["points"])))

        # plt.show()

    def __plot__(self,workflow_id,task_id):
        print "plotting"
        try:
            print "----"
            print self.cluster_alg.clusterResults
            for shape in self.cluster_alg.clusterResults[task_id]:
                for subject_id in self.cluster_alg.clusterResults[task_id][shape]:
                    print subject_id
                    if (len(self.users_per_subject[subject_id]) >= 1):# and (subject_id in self.classification_alg.results):
                        # if self.cluster_alg.clusterResults[task][shape][subject_id]["users"]
                        self.__plot_image__(subject_id)
                        self.__plot_individual_points__(subject_id,task_id,shape)
                        # self.__plot_cluster_results__(subject_id,task,shape)

                        if (self.classification_alg is not None) and (subject_id in self.classification_alg.results):
                            classification_task = "init"
                            classifications = self.classification_alg.results[subject_id][classification_task]
                            # print classifications
                            votes,total = classifications
                            title = self.description[classification_task][0]
                            # print self.description
                            for answer_index,percentage in votes.items():
                                if title != "":
                                    title += "\n"
                                title += self.description[classification_task][answer_index+1] + ": " + str(int(percentage*total))
                            # print  self.description[classification_task][0]
                            # print title

                            plt.title(title)
                        plt.title("number of users: " + str(len(self.users_per_subject[subject_id][task_id])))
                        plt.savefig("/home/greg/Databases/"+self.project_short_name+"/markings/"+str(subject_id)+".jpg")
                        plt.close()
                        # assert False
        except KeyError as e:
            print self.cluster_alg.clusterResults.keys()
            raise

    def __postgres_connect(self,database_details):

        database = database_details[self.environment]["database"]
        username = database_details[self.environment]["username"]
        password = database_details[self.environment]["password"]
        host = database_details[self.environment]["host"]

        # try connecting to the db
        details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
        for i in range(20):
            try:
                self.postgres_session = psycopg2.connect(details)
                self.postgres_cursor = self.postgres_session.cursor()
                break
            except psycopg2.OperationalError as e:
                pass

        if self.postgres_session is None:
            raise psycopg2.OperationalError()

        # select = "SELECT * from project_contents"
        # cur = self.postgres_session.cursor()
        # cur.execute(select)

    def __postgres_backup__(self):
        local_session = psycopg2.connect("dbname='tate' user='panoptes' host='localhost' password='apassword'")
        local_cursor = local_session.cursor()

        # local_cursor.execute("CREATE TABLE annotate_classifications (project_id integer,user_id integer,workflow_id integer,annotations text,created_at timestamp,updated_at timestamp,user_group_id integer, user_ip inet, completed boolean, gold_standard boolean, expert_classifier integer, metadata text, subject_ids integer, workflow_version text);")
        local_cursor.execute("CREATE TABLE annotate_classifications (annotations json);")

        select = "SELECT * from classifications where project_id="+str(self.project_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)

        for ii,t in enumerate(cur.fetchall()):
            id_,project_id,user_id,workflow_id,annotations,created_at,updated_at,user_group_id,user_ip,completed,gold_standard,expert_classifier,metadata,subject_ids,workflow_version = t
            print type(json.dumps(annotations))
            # local_cursor.execute("""
            #     INSERT INTO annotate_classifications
            #     (project_id,user_id,workflow_id,annotations,created_at,updated_at,user_group_id, user_ip, completed, gold_standard, expert_classifier, metadata, subject_id, workflow_version)
            #     values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            #     (id_,project_id,user_id,workflow_id,json.dumps(annotations),created_at,updated_at,user_group_id,user_ip,completed,gold_standard,expert_classifier,json.dumps(metadata),subject_ids[0],workflow_version))
            st = json.dumps(annotations)
            local_cursor.execute("""
                INSERT INTO annotate_classifications
                (annotations)
                values (%s)""",
                (annotations))


    def __readin_tasks__(self,workflow_id):
        """
        get the details for each task - for example, what tasks might we want to run clustering algorithms on
        and if so, what params related to that task are relevant
        :return:
        """
        # get the tasks associated with the given workflow
        select = "SELECT tasks from workflows where id = " + str(workflow_id)
        self.postgres_cursor.execute(select)
        tasks = self.postgres_cursor.fetchone()[0]

        # select = "SELECT * from workflow_contents where id = " + str(workflow_id)
        # self.postgres_cursor.execute(select)
        # print
        # print self.postgres_cursor.fetchone()

        # which of these tasks have classifications associated with them?
        classification_tasks = {}
        # which have drawings associated with them
        marking_tasks = {}

        for task_id in tasks:
            # self.task_type[task_id] = tasks[task_id]["type"]

            # if the task is a drawing one, get the necessary details for clustering
            if tasks[task_id]["type"] == "drawing":
                marking_tasks[task_id] = []
                # manage marking tools by the marking type and not the index
                # so all ellipses will be clustered together

                # # see if mulitple tools are creating the same shape
                # counter = {}

                for tool in tasks[task_id]["tools"]:
                    # shape = ellipse, line, pt etc.
                    shape = tool["type"]

                    # extract the label of the tool - this means that things don't have to ordered
                    label = tool["label"]
                    label_words = label.split(".")
                    tool_id = int(label_words[2])

                    # are there any classification questions associated with this marking?

                    if ("details" in tool) and (tool["details"] is not None) and (tool["details"] != []):
                        if task_id not in classification_tasks:
                            classification_tasks[task_id] = {}
                        if "subtask" not in classification_tasks[task_id]:
                            classification_tasks[task_id]["subtask"] = {}
                        if tool_id not in classification_tasks[task_id]["subtask"]:
                            classification_tasks[task_id]["subtask"][tool_id] = range(len(tool["details"]))
                        # if tool_id not in self.classification_tasks[task_id]:
                        #     self.classification_tasks[task_id][tool_id] = {}
                        # classification_tasks[task_id][tool_id]= [i for i in range(len(tool["details"]))]
                        # todo - fix this

                    print "tool is " + tool["type"]
                    if tool["type"] == "line":
                        marking_tasks[task_id].append("line")
                    elif tool["type"] == "ellipse":
                        marking_tasks[task_id].append("ellipse")
                    elif tool["type"] == "point":
                        marking_tasks[task_id].append("point")
                    elif tool["type"] == "circle":
                        marking_tasks[task_id].append("circle")
                    elif tool["type"] == "rectangle":
                        marking_tasks[task_id].append("rectangle")
                    elif tool["type"] == "polygon":
                        marking_tasks[task_id].append("polygon")
                    else:
                        print tool
                        assert False

            else:
                # self.marking_params_per_task[task_id] = []
                classification_tasks[task_id] = True

        # find out if any of the shapes for a given task are "confusing"
        # that is more, there is more than 1 tool which can create that shape
        for task_id in marking_tasks:
            for shape in ["line","ellipse","point","circle","rectangle"]:
                if sum([1 for s in marking_tasks[task_id] if s == shape]) > 1:
                    # this shape is confusing
                    if task_id not in classification_tasks:
                        classification_tasks[task_id] = {}
                    if "shapes" not in classification_tasks[task_id]:
                        classification_tasks[task_id]["shapes"] = []

                    classification_tasks[task_id]["shapes"].append(shape)

        # print workflow_id
        # print tasks
        # assert False
        return classification_tasks,marking_tasks

    def __remove_user_ids__(self,aggregation):
        """
        ids are needed for aggregation but they shouldn't be stored with the results
        NOTE ids are postgres ids, NOT ip or email addresses
        """
        for subject_id in aggregation:
            if subject_id == "param":
                continue

            for task_id in aggregation[subject_id]:
                if task_id == "param":
                    continue
                if isinstance(aggregation[subject_id][task_id],dict):
                    for shape in aggregation[subject_id][task_id]:
                        if shape == "param":
                            continue

                        for cluster_index in aggregation[subject_id][task_id][shape]:
                            if cluster_index == "param":
                                continue

                            assert isinstance(aggregation[subject_id][task_id][shape][cluster_index],dict)
                            #aggregation[subject_id][task_id][shape][cluster_index].pop("users",None)

                            del aggregation[subject_id][task_id][shape][cluster_index]["users"]

        return aggregation

    def __set_classification_alg__(self,alg):
        self.classification_alg = alg
        assert isinstance(self.classification_alg,classification.Classification)

    def __set_clustering_algs__(self,clustering_algorithms):
        # the dictionary allows us to give a different clustering algorithm for different shapes

        self.cluster_algs = {}
        assert isinstance(clustering_algorithms,dict)
        for kw in clustering_algorithms:
            assert kw in self.marking_params_per_shape
            self.cluster_algs[kw] = clustering_algorithms[kw](kw)
            assert isinstance(self.cluster_algs[kw],clustering.Cluster)

    def __setup_workflows__(self):#,project_id):
        request = urllib2.Request(self.host_api+"workflows?project_id="+str(self.project_id))
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)

        # request
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError as e:
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
            print 'Error response body: ', e.read()
            raise
        except urllib2.URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
            raise
        else:
            # everything is fine
            body = response.read()

        # put it in json structure and extract id
        data = json.loads(body)

        # for each workflow, read in the tasks
        workflows = {}


        for individual_work in data["workflows"]:
            id_ = int(individual_work["id"])
            # self.workflows.append(id_)
            workflows[id_] = self.__readin_tasks__(id_)
            # self.subject_sets[id_] = self.__get_subject_ids__(id_)

        # assert False
        # read in the most current version of each of the workflows
        return workflows

    def __sort_classifications__(self,workflow_id,subject_set=None):
        version = int(math.floor(float(self.versions[workflow_id])))

        classification_tasks,marking_tasks = self.workflows[workflow_id]
        raw_classifications = {}

        if subject_set is None:
            subject_set = self.__load_subjects__(workflow_id)

        total = 0
        for s in self.__chunks(subject_set,15):
            statements_and_params = []
            ignore_version = True

            select_statement = self.cassandra_session.prepare("select user_id,annotations,workflow_version from "+self.classification_table+" where project_id = ? and subject_id = ? and workflow_id = ?")# and workflow_version = ?")
            for subject_id in s:
                params = (int(self.project_id),subject_id,int(workflow_id))#,version)
                statements_and_params.append((select_statement, params))
            results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=False)

            for subject_id,(success,record_list) in zip(s,results):
                print record_list
                if not success:
                    print record_list
                    assert success


                # to help us tell apart between different users who are not logged in
                # todo- a non logged in user might see this subject multiple times - how to protect against that?
                non_logged_in_users = 0
                # print "==++ " + str(subject_id)
                for record in record_list:
                    total += 1
                    user_id = record.user_id
                    if user_id == -1:
                        non_logged_in_users += -1
                        user_id = non_logged_in_users

                    annotations = json.loads(record.annotations)
                    # go through each annotation and get the associated task
                    for task in annotations:
                        task_id = task["task"]

                        # is this a classification task
                        if task_id in classification_tasks:
                            # is this classification task associated with a marking task?
                            # i.e. a sub task?
                            if isinstance(classification_tasks[task_id],dict):
                                # does this correspond to a confusing shape?
                                # print classification_tasks[task_id]
                                for marking in task["value"]:
                                    tool = marking["tool"]
                                    shape = marking_tasks[task_id][tool]

                                    # is this shape confusing?
                                    if ("shapes" in classification_tasks[task_id]) and (shape in classification_tasks[task_id]["shapes"]):
                                        if task_id not in raw_classifications:
                                            raw_classifications[task_id] = {}
                                        if shape not in raw_classifications[task_id]:
                                            raw_classifications[task_id][shape] = {}
                                        if subject_id not in raw_classifications[task_id][shape]:
                                            raw_classifications[task_id][shape][subject_id] = {}

                                        # todo - FIX!!!
                                        try:
                                            relevant_params = self.marking_params_per_shape[shape](marking,(10000,10000))
                                            # assert (relevant_params,user_id) not in raw_classifications[task_id][shape][subject_id]
                                            raw_classifications[task_id][shape][subject_id][(relevant_params,user_id)] = tool #.append((user_id,relevant_params,tool))
                                        except InvalidMarking as e:
                                            print e

                                    # is there a subtask associated with this marking?
                                    # keep in mind that subtasks are by tool type - NOT by shape
                                    # so different tools with the same shape can have different subtasks
                                    # so the below code is slightly different than above
                                    if ("subtask" in classification_tasks[task_id]) and (tool in classification_tasks[task_id]["subtask"]):
                                        for local_subtask_id in classification_tasks[task_id]["subtask"][tool]:
                                            global_subtask_id = str(task_id)+"_"+str(tool)+"_"+str(local_subtask_id)
                                            if global_subtask_id not in raw_classifications:
                                                raw_classifications[global_subtask_id] = {}
                                            if subject_id not in raw_classifications[global_subtask_id]:
                                                raw_classifications[global_subtask_id][subject_id] = {}

                                            # we need the coordinates since different markings may have the same subtasks
                                            shape = marking_tasks[task_id][tool]
                                            relevant_params = self.marking_params_per_shape[shape](marking,(10000,10000))

                                            subtask_value = marking["details"][local_subtask_id]["value"]
                                            raw_classifications[global_subtask_id][subject_id][(relevant_params,user_id)] = subtask_value


                            else:
                                if task_id not in raw_classifications:
                                    raw_classifications[task_id] = {}
                                if subject_id not in raw_classifications[task_id]:
                                    raw_classifications[task_id][subject_id] = []
                                # if task_id == "init":
                                #     print task_id,task["value"]
                                raw_classifications[task_id][subject_id].append((user_id,task["value"]))

        return raw_classifications

    def __sort_markings__(self,workflow_id,subject_set=None,ignore_version=False):
        """
        :param workflow_id:
        :param subject_set:
        :param ignore_version: ONLY set for debugging use
        :return:
        """
        ignore_version=False

        # workflow_version = self.__get_workflow_version__(workflow_id)

        classification_tasks,marking_tasks = self.workflows[workflow_id]
        if marking_tasks == {}:
            return {}

        raw_markings = {}
        if subject_set is None:
            subject_set = self.__load_subjects__(workflow_id)

        # use one image from the workflow to determine the size of all images
        # todo - BAD ASSUMPTION, think of something better
        try:
            fname = self.__image_setup__(subject_set[0])
            im=Image.open(fname)
            width,height= im.size
        except (IndexError,ImageNotDownloaded,AttributeError):
            # todo - fix!!!
            print "image not downloaded"
            width = 1000
            height = 1000

        self.users_per_subject={}

        loaded_subjects = set()
        read_in = set()

        for s in self.__chunks(subject_set,15):
            print s
            statements_and_params = []
            if ignore_version:
                select_statement = self.cassandra_session.prepare("select * from " + self.classification_table + " where project_id = ? and workflow_id = ? and subject_id = ?")# and workflow_id = ?")# and workflow_version = ?")
            else:
                select_statement = self.cassandra_session.prepare("select * from " + self.classification_table + " where project_id = ? and workflow_id = ? and subject_id = ? and workflow_version = ?")# and workflow_id = ?")# and workflow_version = ?")

            # select_statement = self.cassandra_session.prepare("select id,user_id,annotations from classifications where project_id = ? and subject_id = ? and workflow_id = ?")
            # select_statement = self.cassandra_session.prepare("select * from classifications where project_id = ? and workflow_id = ? and workflow_version = ? and subject_id = ?")# and workflow_id = ?")# and workflow_version = ?")
            # select_statement = self.cassandra_session.prepare("select * from classifications where project_id = ?")# and workflow_id = ? and workflow_version = ?")

            # assert 458701 not in s
            # print s

            for subject_id in s:
                if ignore_version:
                    params = (int(self.project_id),workflow_id,subject_id,)#int(workflow_id))#,int(math.floor(float(self.versions[workflow_id]))))
                else:
                    # version = int(math.floor(float(self.versions[workflow_id])))
                    params = (int(self.project_id),workflow_id,subject_id,self.versions[workflow_id])#int(workflow_id))#,int(math.floor(float(self.versions[workflow_id]))))
                # params = (int(self.project_id),)#,int(workflow_id),int(math.floor(float(self.versions[workflow_id]))))
                statements_and_params.append((select_statement, params))
                # print params
            results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=False)
            for subject_id,(success,record_list) in zip(s,results):
                tt = 0
                # todo - implement error recovery
                if not success:
                    print record_list
                assert success

                # use this counter to help differentiate non logged in users
                non_logged_in_users = 0

                # print (success,record_list)

                ips_per_subject = []

                for record in record_list:
                    tt += 1
                    user_id = record.user_id

                    if record.user_ip in ips_per_subject:
                        # print "duplicate ip address per subject"
                        continue

                    ips_per_subject.append(record.user_ip)

                    if user_id == -1:
                        non_logged_in_users += -1
                        user_id = non_logged_in_users
                    loaded_subjects.add(subject_id)
                    # for counting the number of users who have seen this subject
                    # set => in case someone has seen this image twice
                    if subject_id not in self.users_per_subject:
                        self.users_per_subject[subject_id] = {}

                    # # todo - how to handle cases where someone has seen an image more than once?
                    # if user_id in self.users_per_subject[subject_id]:
                    #     select = "SELECT * from users where id="+str(user_id)
                    #     cur = self.postgres_session.cursor()
                    #     cur.execute(select)
                    #
                    #     print cur.fetchone()

                    # self.users_per_subject[subject_id].add(user_id)

                    annotations = json.loads(record.annotations)

                    # go through each annotation and get the associated task
                    for task in annotations:
                        task_id = task["task"]

                        if task_id not in self.users_per_subject[subject_id]:
                            self.users_per_subject[subject_id][task_id] = set()
                        self.users_per_subject[subject_id][task_id].add(user_id)

                        if task_id in marking_tasks:
                            if not isinstance(task["value"],list):
                                print "not properly formed marking - skipping"
                                continue

                            for marking in task["value"]:
                                # what kind of tool made this marking and what was the shape of that tool?
                                try:
                                    tool = marking["tool"]
                                    shape = marking_tasks[task_id][tool]
                                except KeyError:
                                    tool = None
                                    shape = marking["type"]
                                except IndexError as e:
                                    print marking
                                    print marking_tasks
                                    print task_id
                                    print tool
                                    raise
                                if shape ==  "image":
                                    # todo - treat image like a rectangle
                                    continue

                                if shape not in self.marking_params_per_shape:
                                    print "unrecognized shape: " + shape
                                    continue

                                try:
                                    # extract the params specifically relevant to the given shape
                                    relevant_params = self.marking_params_per_shape[shape](marking,(width,height))

                                    # only create these if we have a valid marking
                                    if task_id not in raw_markings:
                                        raw_markings[task_id] = {}
                                    if shape not in raw_markings[task_id]:
                                        raw_markings[task_id][shape] = {}
                                    if subject_id not in raw_markings[task_id][shape]:
                                        raw_markings[task_id][shape][subject_id] = []

                                    raw_markings[task_id][shape][subject_id].append((user_id,relevant_params,tool))
                                except InvalidMarking as e:
                                    # print e
                                    pass

        return raw_markings

    def __store_results__(self,workflow_id,aggregations):
        # aggregations = self.__remove_user_ids__(aggregations)
        # cur = self.postgres_session.cursor()
        # finally write the results into the postgres db

        # subject_set = self.__load_subjects__(workflow_id)
        # assert sorted(self.aggregations.keys()) == sorted(subject_set)

        # stmt = "SELECT * from aggregations"
        # cur.execute(stmt)

        for ii,subject_id in enumerate(aggregations):
            if subject_id == "param":
                continue
            if (ii > 0) and (ii % 10000 == 0):
                self.postgres_session.commit()

            # skip if we don't have any aggregation results yet

            # there have been requests for the aggregation to also contain the metadata
            select = "SELECT metadata from subjects where id="+str(subject_id)
            print "inserting subject id " + str((workflow_id,subject_id))
            self.postgres_cursor.execute(select)
            metadata = self.postgres_cursor.fetchone()

            # add in the necessary metadata relevant to this subject - probably filename etc.
            # and the instructions - so people can match up the keys with what the actual tasks were
            # slightly redundant since this will not change for a workflow but will be given for every subject
            aggregation = aggregations[subject_id]
            aggregation["metadata"] = metadata
            aggregation["instructions"] = self.__get_workflow_details__(workflow_id)
            stmt = "INSERT INTO aggregations(workflow_id,subject_id,aggregation,created_at,updated_at) VALUES("+str(workflow_id)+","+str(subject_id)+",'"+json.dumps(aggregation)+"','"+str(datetime.datetime.now())+"','"+str(datetime.datetime.now())+"')"
            self.postgres_cursor.execute(stmt)
        print "^^^^"
        self.postgres_session.commit()

    def __upsert_results__(self,workflow_id,aggregations):
        # postgres sucks
        self.postgres_cursor.execute("select subject_id from aggregations where workflow_id = " + str(workflow_id))
        r = [i[0] for i in self.postgres_cursor.fetchall()]
        print r

        for subject_id in aggregations:
            if subject_id == "param":
                continue
            select = "SELECT metadata from subjects where id="+str(subject_id)
            print "inserting subject id " + str((workflow_id,subject_id))
            self.postgres_cursor.execute(select)
            metadata = self.postgres_cursor.fetchone()

            aggregation = aggregations[subject_id]
            aggregation["metadata"] = metadata
            aggregation["instructions"] = self.__get_workflow_details__(workflow_id)

            if subject_id in r:
                # do an update
                # del aggregation["instructions"]
                t = json.dumps(aggregation)
                # t = re.sub("\"","'",t)
                # stmt = "UPDATE aggregations SET aggregation = \""+t+"\" WHERE workflow_id = " + str(workflow_id) + " and subject_id = " + str(subject_id)
                # stmt = "UPDATE aggregations (aggregation,updated_at) VALUES(" + json.dumps(aggregation) + "," + str(datetime.datetime.now()) + ") WHERE workflow_id = " + str(workflow_id) + " and subject_id = " + str(subject_id)
                stmt = "UPDATE \"aggregations\" SET \"aggregation\" = \'"+t+"\' WHERE \"aggregations\".\"workflow_id\" = " + str(workflow_id) + " and \"aggregations\".\"subject_id\" = " + str(subject_id)
            else:
                # do an insert
                stmt = "INSERT INTO aggregations(workflow_id,subject_id,aggregation,created_at,updated_at) VALUES("+str(workflow_id)+","+str(subject_id)+",'"+json.dumps(aggregation)+"','"+str(datetime.datetime.now())+"','"+str(datetime.datetime.now())+"')"
            # print stmt
            self.postgres_cursor.execute(stmt)

        self.postgres_session.commit()


class SubjectGenerator:
    def __init__(self,project):
        self.project = project

    def __iter__(self):
        subject_ids = []
        for subject in self.project.subject_collection.find():
            subject_ids.append(subject["zooniverse_id"])

            if len(subject_ids) == 50:
                yield subject_ids
                subject_ids = []

        yield  subject_ids
        raise StopIteration

def twod_linesegment(pt):
    x1,x2,y1,y2 = pt
    print x1,x2,y1,y2
    dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

    try:
        tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
        theta = math.atan(tan_theta)
    except ZeroDivisionError:
        theta = math.pi/2.

    return dist,theta

if __name__ == "__main__":
    project_name = sys.argv[1]
    project = AggregationAPI(project_name)

    # project.__migrate__()

    project.__set_clustering_algs__({"point":agglomerative.Agglomerative,"rectangle":blob_clustering.BlobClustering})#, "rectangle":(blob_clustering.BlobClustering,{})})
    project.__set_classification_alg__(classification.VoteCount())
    # project.__info__()
    # project.__aggregate__()#workflows=[84],subject_set=[495225])#subject_set=[460208, 460210, 460212, 460214, 460216])
    project.__get_results__(979)
    # project.__get_workflow_details__(84)
