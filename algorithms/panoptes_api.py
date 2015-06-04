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
import urllib
import datetime

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

class PanoptesAPI:
    #@profile
    def __init__(self,project):#,user_threshold= None, score_threshold= None): #Supernovae
        # self.user_threshold = user_threshold
        # first find out which environment we are working with
        self.environment = os.getenv('ENVIRONMENT', "production")
        print self.environment

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
        self.token = None

        # the http api for connecting to Panoptes
        self.http_api = None
        print "connecting to Panoptes http api"

        # set the http_api and basic project details
        self.__panoptes_connect__()

        # details about the project we are going to work with
        self.project_id = self.__get_project_id()
        # get the most recent workflow version
        self.workflow_version = self.__get_workflow_version()
        self.workflow_id = self.__get_workflow_id()

        # now connect to the Panoptes db - postgres
        try:
            database_file = open("config/database.yml")
        except IOError:
            database_file = open(base_directory+"/Databases/database.yml")
        database_details = yaml.load(database_file)
        self.conn = None
        self.__postgres_connect(database_details)

        # and to the cassandra db as well
        self.__cassandra_connect()

        self.task_type = {}
        self.cluster_params_per_task = {}

    def __postgres_connect(self,database_details):

        database = database_details[self.environment]["database"]
        username = database_details[self.environment]["username"]
        password = database_details[self.environment]["password"]
        host = database_details[self.environment]["host"]

        # try connecting to the db
        details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
        for i in range(20):
            try:
                self.conn = psycopg2.connect(details)
                break
            except psycopg2.OperationalError as e:
                pass

        if self.conn is None:
            raise psycopg2.OperationalError()

    def __cassandra_connect(self):
        """
        connect to the AWS instance of Cassandra - try 10 times and raise an error
        :return:
        """
        for i in range(10):
            try:
                self.cluster = Cluster(['panoptes-cassandra.zooniverse.org'])
                self.session = self.cluster.connect('demo')
                return
            except cassandra.cluster.NoHostAvailable:
                pass

        assert False

    def __subject_set_details__(self):
        select = "SELECT * from media"
        cur = self.conn.cursor()
        cur.execute(select)
        subject_set = cur.fetchone()
        print subject_set



    def __task_setup__(self):
        """
        get the details for each task - for example, what tasks might we want to run clustering algorithms on
        and if so, what params related to that task are relevant
        :return:
        """
        select = "SELECT * from workflows where project_id = " + str(self.project_id)
        cur = self.conn.cursor()
        cur.execute(select)
        tasks = cur.fetchone()[2]
        print tasks
        for task_id in tasks:

            # print tasks[task_id]
            self.task_type[task_id] = tasks[task_id]["type"]
            if tasks[task_id]["type"] == "drawing":
                self.cluster_params_per_task[task_id] = []
                for tool in tasks[task_id]["tools"]:
                    if tool["type"] == "line":
                        self.cluster_params_per_task[task_id].append(("x1","x2","y1","y2"))
                    else:
                        assert False
            else:
                self.cluster_params_per_task[task_id] = []

        print self.cluster_params_per_task

    def __get_markings__(self,subject_id,gold_standard):
        # current_subject_id = None
        # project_id
        # users_per_drawing_task = {}
        # for each drawing task, we can
        # get all of the drawing task ids
        # drawing_task_ids = [task_id for task_id in self.task_type if self.task_type[task_id] == "drawing"]
        # create a list for each tool and task pair
        #drawing_tasks = {task_id:[[] for tool in self.cluster_params_per_task[task_id] ] for task_id in drawing_task_ids}
        #print drawing_tasks

        drawings = {}

        # go through each of the classifications for the given subject
        for classification in self.session.execute("select user_id,annotations from classifications where project_id = " + str(self.project_id) + " and subject_id = " + str(subject_id)):
            # convert from string into json
            annotations = json.loads(classification.annotations)

            # go through each annotation, looking for markings
            for task in annotations:
                task_id = task["task"]

                # if we have found a list of markings, associated each one with the tool that made it
                if self.task_type[task_id] == "drawing":
                    for drawing in task["value"]:
                        # get the index of the tool that made the marking
                        tool = drawing["tool"]
                        drawing_params = self.cluster_params_per_task[task_id][drawing["tool"]]
                        # extract the params that are relevant to the marking
                        relevant_params = [drawing[p] for p in drawing_params]

                        # todo: fix 0 = frame
                        id_ = (task_id,0,tool)
                        if id_ not in drawings:
                            drawings[id_] = [(classification.user_id,relevant_params)]
                        else:
                            drawings[id_].append((classification.user_id,relevant_params))

        return drawings

    def __migrate__(self):
        try:
            self.session.execute("drop table classifications")
        except cassandra.InvalidRequest:
            pass
        # self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, metadata text, subject_id int, workflow_version text, PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
        self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, subject_id int, workflow_version float,metadata text, PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
        # self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int,user_ip inet,subject_id int,annotations text,  PRIMARY KEY(project_id,subject_id,user_id,user_ip) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")

        select = "SELECT * from classifications where project_id="+str(self.project_id)+" and workflow_id=" + str(self.workflow_id)
        cur = self.conn.cursor()
        cur.execute(select)

        for ii,t in enumerate(cur.fetchall()):
            print ii
            id_,project_id,user_id,workflow_id,annotations,created_at,updated_at,user_group_id,user_ip,completed,gold_standard,expert_classifier,metadata,subject_ids,workflow_version = t
            if gold_standard != True:
                gold_standard = False

            if not isinstance(user_group_id,int):
                user_group_id = -1

            if not isinstance(user_id,int):
                user_id = -1

            self.session.execute(
                """
                insert into classifications (project_id, user_id, workflow_id, annotations, created_at, updated_at, user_group_id, user_ip,  completed, gold_standard, subject_id, workflow_version,metadata)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (project_id, user_id, workflow_id, json.dumps(annotations), created_at, updated_at, user_group_id, user_ip,  completed, gold_standard,  subject_ids[0], float(workflow_version),json.dumps(metadata)))


    def __get_workflow_id(self):#,project_id):
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
        #print data["workflows"]
        return int(data["workflows"][0]['id'])

    def __get_project_id(self):
        """
        get the id number for our project
        :return:
        """
        request = urllib2.Request(self.host_api+"projects?owner="+urllib2.quote(self.owner)+"&display_name="+urllib2.quote(self.project_name))
        # request = urllib2.Request(self.host_api+"projects?owner="+self.owner+"&display_name=Galaxy%20Zoo%20Bar%20Lengths")
        # print hostapi+"projects?owner="+owner+"&display_name="+project_name
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)

        # request
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError as e:
            print self.host_api+"projects?owner="+self.owner+"&display_name="+self.project_name
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
            print 'Error response body: ', e.read()
        except urllib2.URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
        else:
            # everything is fine
            body = response.read()

        # put it in json structure and extract id
        data = json.loads(body)
        return data["projects"][0]["id"]

    def __get_workflow_version(self):#,project_id):
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
        except urllib2.URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
        else:
            # everything is fine
            body = response.read()

        # put it in json structure and extract id
        data = json.loads(body)
        return data["workflows"][0]['version']

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
                devise_login_data=("{\"user\": {\"display_name\":\""+self.user_name+"\",\"password\":\""+self.password+
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


    def __postgres_connect__(self):
        # print "workflow id is " + str(workflow_id)
        print "connecting to Panoptes database"
        # now load in the details for accessing the database
        try:
            database_file = open("config/database.yml")
        except IOError:
            database_file = open(base_directory+"/Databases/database.yml")
        database_details = yaml.load(database_file)

        #environment = "staging"

        database = database_details[self.environment]["database"]
        username = database_details[self.environment]["username"]
        password = database_details[self.environment]["password"]
        host = database_details[self.environment]["host"]

        # try connecting to the db
        self.conn = None
        details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
        for i in range(20):
            try:
                self.conn = psycopg2.connect(details)
                break
            except psycopg2.OperationalError as e:
                pass

        if self.conn is None:
            raise psycopg2.OperationalError()

    def __set_clustering_alg__(self,clustering_alg):
        self.cluster_alg = clustering_alg(self)

    def __cluster__(self,subject_id):
        self.cluster_alg.__fit__(subject_id)

    def __store_results__(self,subject_id):
        try:
            self.session.execute("drop table aggregations")
        except cassandra.InvalidRequest:
            pass
        # self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, metadata text, subject_id int, workflow_version text, PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
        self.session.execute("CREATE TABLE aggregations (project_id int, workflow_id int, subject_id int, task text, frame int, tool int, aggregation text, created_at timestamp, updated_at timestamp, PRIMARY KEY(project_id,workflow_id,subject_id) ) WITH CLUSTERING ORDER BY (workflow_id ASC, subject_id ASC);")

        for (task,frame,tool),aggregation in self.cluster_alg.clusterResults[subject_id].items():
            self.session.execute(
                """
                insert into aggregations (project_id, workflow_id, subject_id, task, frame, tool, aggregation, created_at, updated_at)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (int(self.project_id),self.workflow_id,subject_id,task,frame,tool,json.dumps(aggregation),datetime.datetime.now(),datetime.datetime.now()))
                # (project_id, user_id, workflow_id, json.dumps(annotations), created_at, updated_at, user_group_id, user_ip,  completed, gold_standard,  subject_ids[0], float(workflow_version),json.dumps(metadata)))



brooke = PanoptesAPI("bar_lengths")
brooke.__task_setup__()
# brooke.__get_markings__(3266)

import agglomerative
brooke.__set_clustering_alg__(agglomerative.Agglomerative)
# a = agglomerative.Agglomerative(brooke)
brooke.__cluster__(3266)
brooke.__store_results__(3266)