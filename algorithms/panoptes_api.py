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
import classification
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.patches import Ellipse
from clustering import  cnames
import numpy

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

class PanoptesAPI:
    #@profile
    def __init__(self,project=None):#,user_threshold= None, score_threshold= None): #Supernovae
        # # self.user_threshold = user_threshold
        # # first find out which environment we are working with
        # self.environment = os.getenv('ENVIRONMENT', "staging")
        # self.environment = "staging"

        if project is not None:
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
            self.project_id = self.__get_project_id()
            # get the most recent workflow version
            # todo - probably need to make sure that we only read in classifications from the most recent workflow version
            self.workflow_version = self.__get_workflow_version()
            self.workflow_id = self.__get_workflow_id()



        # now connect to the Panoptes db - postgres
        try:
            database_file = open("config/database.yml")
        except IOError:
            database_file = open(base_directory+"/Databases/database.yml")

        database_details = yaml.load(database_file)
        self.postgres_session = None

        self.__postgres_connect(database_details)
        # select = "SELECT * from projects"
        # cur = self.postgres_session.cursor()
        # cur.execute(select)
        # for project in cur.fetchall():
        #     print project
        # assert False

        # and to the cassandra db as well
        if project is not None:
            print self.workflow_id
            self.__cassandra_connect()

            self.task_type = {}
            # used to extract the relevant parameters from a marking
            self.marking_params_per_task = {}
            # used to store the shapes per tool - useful for when we want to plot out results
            # not strictly necessary but should make things a lot easier to understand/follow
            self.shapes_per_tool = {}

            # cluster algorithm to be used, if any
            self.cluster_alg = None
            self.classification_alg = None



        self.workflows = {}
        self.__load_workflows__()
        # if there is only one workflow for the project, just use that
        # otherwise, we will need to specify which workflow we want to use
        self.workflow_id = None
        # subjects don't have to be the same for all workflows for the same project - so only load the subjects
        # if there workflow is given (e.g. there is only one workflow)
        self.subjects = []
        if len(self.workflows) == 1:
            self.workflow_id = self.workflows.keys()[0]

            # load in the subjects for the project
            self.__get_subject_ids__()

            # load the tasks associated with this workflow
            self.__task_setup__()

        self.classifications = None
        self.aggregations = None

    def __set_workflow__(self,workflow_id):
        """
        set the workflow id - must be an actual workflow associated with this project
        :param workflow_id:
        :return:
        """
        assert workflow_id in self.workflows
        self.workflow_id = workflow_id

        # load in the subjects specific to that workflow
        self.__get_subject_ids__()

        # load the tasks associated with this workflow
        self.__task_setup__()

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


    def __cassandra_connect(self):
        """
        connect to the AWS instance of Cassandra - try 10 times and raise an error
        :return:
        """
        for i in range(10):
            try:
                self.cluster = Cluster(['panoptes-cassandra.zooniverse.org'])
                self.cassandra_session = self.cluster.connect('zooniverse')
                return
            except cassandra.cluster.NoHostAvailable:
                pass

        assert False

    def __get_subject_ids__(self):
        """
        return the subjects associated with a given workflow (which needs to be specific for the project)
        :return:
        """
        # the workflow_id needs to have already been set
        assert self.workflow_id is not None
        self.subjects = []

        # there may be multiple subject sets associated with each workflow - all of them are relevant
        select = "SELECT subject_set_id from subject_sets_workflows where workflow_id = " + str(self.workflow_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)
        for subject_set_id in cur.fetchall():

            select = "SELECT subject_id FROM set_member_subjects WHERE subject_set_id = " + str(subject_set_id[0])
            cur2 = self.postgres_session.cursor()
            cur2.execute(select)
            subjects = cur2.fetchall()
            self.subjects.extend([s[0] for s in subjects])
        print self.subjects
        assert self.subjects != []


    def __task_setup__(self):
        """
        get the details for each task - for example, what tasks might we want to run clustering algorithms on
        and if so, what params related to that task are relevant
        :return:
        """
        select = "SELECT tasks from workflows where id = " + str(self.workflow_id)
        # cur = self.postgres_session.cursor()
        self.postgres_cursor.execute(select)
        tasks = self.postgres_cursor.fetchone()[0]
        self.classification_tasks = {}

        print tasks
        print
        for task_id in tasks:

            # print tasks[task_id]
            self.task_type[task_id] = tasks[task_id]["type"]

            # if the task is a drawing one, get the necessary details for clustering
            if tasks[task_id]["type"] == "drawing":

                self.marking_params_per_task[task_id] = []
                self.shapes_per_tool[task_id] = []
                # manage marking tools by the marking type and not the index
                # so all ellipses will be clustered together

                for tool in tasks[task_id]["tools"]:
                    # shape = ellipse, line, pt etc.
                    shape = tool["type"]
                    # extract the label of the tool - this means that things don't have to ordered
                    label = tool["label"]
                    label_words = label.split(".")
                    tool_id = int(label_words[2])

                    # if shape not in self.classification_tasks[task_id]:
                    #     # store which tools in this task actually make this particular shape
                    #     self.classification_tasks[task_id][shape] = {"tools":[tool_id]}
                    # else:
                    #     self.classification_tasks[task_id][shape]["tools"].append(tool_id)



                    # # are there questions associated with the marking?
                    # # is there another tool with the same shape with questions? - if so, the questions need to be
                    # # the same
                    # if "details" in self.classification_tasks[task_id][shape]:
                    #     # todo - fill this in
                    #     assert False
                    # else:
                    # are there any classification questions associated with this marking?
                    # todo - might need to check if tool["details"] != [] or something like that
                    if tool["details"] is not None:
                        if task_id not in self.classification_tasks:
                            self.classification_tasks[task_id] = {}
                        # if tool_id not in self.classification_tasks[task_id]:
                        #     self.classification_tasks[task_id][tool_id] = {}
                        self.classification_tasks[task_id][shape]= [i for i in range(len(tool["details"]))]
                    #     # this is the first tool associated with this shape/task we have encountered
                    #
                    #     for question_index,question in enumerate(tool["details"]):
                    #         # the question can be divided into tasks, tool and question index
                    #         # but we should also know the task and tool id
                    #         if task_id not in self.classification_tasks:
                    #             self.classification_tasks[task_id] = {}
                    #
                    #         # if len(self.voting_tasks) <= task_id:
                    #         #     self.voting_tasks.append({})
                    #
                    #
                    #         # if :
                    #         #     self.classification_tasks[task_id][tool_id] = []
                    #         self.classification_tasks[task_id][tool_id].append(question_index)
                    #     # question_id = question["question"].split(".")
                    #     # self.voting_tasks.append((task_id,tool,question_index))
                    #
                    # # print tool["details"] != []
                    # # if "details" in tool:
                    # #     print tool["details"]
                    print "tool is " + tool["type"]
                    if tool["type"] == "line":
                        self.shapes_per_tool[task_id].append("line")
                        self.marking_params_per_task[task_id].append(("x1","x2","y1","y2"))
                    elif tool["type"] == "ellipse":
                        self.shapes_per_tool[task_id].append("ellipse")
                        self.marking_params_per_task[task_id].append(("angle","rx","ry","x","y"))
                    elif tool["type"] == "point":
                        self.shapes_per_tool[task_id].append("point")
                        self.marking_params_per_task[task_id].append(("x","y"))
                    elif tool["type"] == "circle":
                        self.shapes_per_tool[task_id].append("circle")
                        self.marking_params_per_task[task_id].append(("x","y","r"))
                    elif tool["type"] == "rectangle":
                        self.shapes_per_tool[task_id].append("rectangle")
                        self.marking_params_per_task[task_id].append(("x","y","width","height"))
                    else:
                        print tool
                        assert False
            else:
                # self.marking_params_per_task[task_id] = []
                self.classification_tasks[task_id] = True

        print self.classification_tasks
        print self.marking_params_per_task

    def __get_markings__(self,gold_standard=False):
        # current_subject_id = None
        # project_id
        # users_per_drawing_task = {}
        # for each drawing task, we can
        # get all of the drawing task ids
        # drawing_task_ids = [task_id for task_id in self.task_type if self.task_type[task_id] == "drawing"]
        # create a list for each tool and task pair
        # drawing_tasks = {task_id:[[] for tool in self.cluster_params_per_task[task_id] ] for task_id in drawing_task_ids}
        # print drawing_tasks
        assert self.workflow_id is not None
        markings = {}
        for subject_id in self.subjects:
            t_markings = {}

            # use to track non-logged in users
            non_logged_in_users = 0

            # go through each of the classifications for the given subject
            select_stmt = "select user_id,annotations from classifications where project_id = " + str(self.project_id) + " and subject_id = " + str(subject_id) + " and workflow_id = " + str(self.workflow_id)
            for classification in self.cassandra_session.execute(select_stmt):
                user_id = classification.user_id

                # if the user is not logged in, use counter to help differiniate between users
                # todo - if someone views the same image twice, this approach will count both of those
                # todo classifications which is probably not what we want
                if user_id == -1:
                    non_logged_in_users += -1
                    user_id = non_logged_in_users

                # convert from string into json
                annotations = json.loads(classification.annotations)

                # go through each annotation, looking for markings
                for task in annotations:
                    task_id = task["task"]

                    # if we have found a list of markings, associated each one with the tool that made it
                    if self.task_type[task_id] == "drawing":
                        for drawing in task["value"]:
                            # get the index of the tool that made the marking

                            # different markings can use the same tool - so this may have the affect of clustering
                            # across different marking types
                            # see https://github.com/zooniverse/Panoptes-Front-End/issues/525
                            # for discussion
                            tool = drawing["tool"]
                            drawing_params = self.marking_params_per_task[task_id][drawing["tool"]]
                            # extract the params that are relevant to the marking
                            try:
                                relevant_params = [drawing[p] for p in drawing_params]
                            except KeyError:
                                print drawing
                                raise

                            shape = self.shapes_per_tool[task_id][tool]

                            id_ = (task_id,shape)

                            if id_ not in t_markings:
                                t_markings[id_] = [(user_id,relevant_params)]
                            else:
                                t_markings[id_].append((user_id,relevant_params))
            markings[subject_id] = deepcopy(t_markings)
        return markings

    def __migrate__(self):
        try:
            self.cassandra_session.execute("drop table classifications")
            print "table dropped"
        except cassandra.InvalidRequest:
            print "table did not exist"
            pass
        # self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, annotations text, created_at timestamp, updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, metadata text, subject_id int, workflow_version text, PRIMARY KEY(project_id,subject_id,user_id,user_ip,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")
        self.cassandra_session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int, created_at timestamp,annotations text,  updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, subject_id int, workflow_version float,metadata text, PRIMARY KEY(project_id,subject_id,workflow_id,created_at) ) WITH CLUSTERING ORDER BY (subject_id ASC, workflow_id ASC,created_at ASC);")
        # self.session.execute("CREATE TABLE classifications( project_id int, user_id int, workflow_id int,user_ip inet,subject_id int,annotations text,  PRIMARY KEY(project_id,subject_id,user_id,user_ip) ) WITH CLUSTERING ORDER BY (subject_id ASC, user_id ASC);")

        select = "SELECT * from classifications where project_id="+str(self.project_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)

        migrated_subjects = set()

        for ii,t in enumerate(cur.fetchall()):
            # print ii
            id_,project_id,user_id,workflow_id,annotations,created_at,updated_at,user_group_id,user_ip,completed,gold_standard,expert_classifier,metadata,subject_ids,workflow_version = t
            # print annotations
            # print subject_ids

            if gold_standard != True:
                gold_standard = False

            if not isinstance(user_group_id,int):
                user_group_id = -1

            if not isinstance(user_id,int):
                user_id = -1

            migrated_subjects.add(subject_ids[0])

            self.cassandra_session.execute(
                # project_id int, user_id int, workflow_id int, created_at timestamp,annotations text,  updated_at timestamp, user_group_id int, user_ip inet,  completed boolean, gold_standard boolean, subject_id
                """
                insert into classifications (project_id, user_id, workflow_id,  created_at,annotations, updated_at, user_group_id, user_ip, completed, gold_standard, subject_id, workflow_version,metadata)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (project_id, user_id, workflow_id,created_at, json.dumps(annotations), updated_at, user_group_id, user_ip,  completed, gold_standard,  subject_ids[0], float(workflow_version),json.dumps(metadata)))
        # print "migrated " + str(ii) + " classifications "
        # print sorted(list(migrated_subjects))

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

    def __list_projects__(self):
        stmt = "select * from projects"
        self.postgres_cursor.execute(stmt)
        for r in self.postgres_cursor.fetchall():
            print r

    def __get_subject_url_and_fname__(self,subject_id):
        """
        returns the url of the subject image and the file name which you can use to check if the file locally exists
        :param subject_id:
        :return:
        """
        request = urllib2.Request(self.host_api+"subjects/"+str(subject_id))
        request.add_header("Accept","application/vnd.api+json; version=1")
        request.add_header("Authorization","Bearer "+self.token)
        # request
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError as e:
            print self.host_api+"subjects/"+str(subject_id)
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
            print 'Error response body: ', e.read()
        except urllib2.URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
        else:
            # everything is fine
            body = response.read()

        data = json.loads(body)
        url = str(data["subjects"][0]["locations"][0]["image/jpeg"])
        
        slash_index = url.rfind("/")
        fname = url[slash_index+1:]

        return url,fname


    def __get_project_id(self):
        """
        get the id number for our project
        this is mostly a "beta" feature so that given a name of a project, I can find the id
        in practice, aggregation will be run with just giving the project id
        :return:
        """
        # print self.host_api+"projects?owner="+urllib2.quote(self.owner)+"&display_name="+urllib2.quote(self.project_name)
        # assert False
        request = urllib2.Request(self.host_api+"projects?owner="+urllib2.quote(self.owner)+"&display_name="+urllib2.quote(self.project_name))
        request = urllib2.Request(self.host_api+"projects?display_name="+urllib2.quote(self.project_name))
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

        data = json.loads(body)
        # put it in json structure and extract id
        # print data
        return data["projects"][0]["id"]
        # return None

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

    def __set_clustering_alg__(self,clustering_alg):
        self.cluster_alg = clustering_alg(self)

    def __load_workflows__(self):
        select = "SELECT id,display_name from workflows where project_id="+str(self.project_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)

        for record in cur.fetchall():
            self.workflows[record[0]] = record[1]

    def __list_workflows__(self):
        select = "SELECT * from workflows where project_id="+str(self.project_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)

        for record in cur.fetchall():
            print record

    def __print_annotations__(self):
        """
        for debugging - print out the annotations/classifications
        :return:
        """
        select = "SELECT * from classifications where project_id="+str(self.project_id)
        cur = self.postgres_session.cursor()
        cur.execute(select)

        for record in cur.fetchall():
            print record

    def __cluster__(self):
        print "clustering"
        markings = self.__get_markings__()
        print markings
        self.cluster_alg.__aggregate__(markings)

        self.aggregations = self.cluster_alg.clusterResults

    def __plot_individual_points__(self):
        for subject_id in self.subjects:
            print subject_id

            url,fname = self.__get_subject_url_and_fname__(subject_id)

            image_path = base_directory+"/Databases/images/"+fname

            if not(os.path.isfile(image_path)):
                urllib.urlretrieve(url, image_path)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            image_file = cbook.get_sample_data(image_path)
            image = plt.imread(image_file)
            # fig, ax = plt.subplots()
            im = ax.imshow(image)

            markings = self.__get_markings__(subject_id)
            users = []
            for task,shape in markings:
                for user_id,pt in markings[(task,shape)]:
                    if user_id not in users:
                        users.append(user_id)

                    if shape == "line":
                        plt.plot([pt[0],pt[1]],[pt[2],pt[3]],color=cnames.values()[users.index(user_id)])
                    elif shape == "point":
                        plt.plot([pt[0]],[pt[1]],"o",color=cnames.values()[users.index(user_id)])
                    elif shape == "circle":
                        print (pt[0],pt[1]),pt[2]
                        e = Ellipse((pt[0],pt[1]),width = pt[2],height=pt[2],fill=False,color=cnames.values()[users.index(user_id)])
                        # circle = plt.Circle((pt[0],pt[1]),pt[2],color=cnames.values()[users.index(user_id)])
                        plt.gca().add_patch(e)
                        # ax.add_artist(e)
                        # e.set_alpha(0)
                    elif shape == "ellipse":
                        # ("angle","rx","ry","x","y")
                        e = Ellipse((pt[3],pt[4]),width = pt[2],height=pt[1],fill=False,angle=pt[0],color=cnames.values()[users.index(user_id)])
                    elif shape == "rectangle":
                        plt.plot([pt[0],pt[0]+pt[2]],[pt[1],pt[1]],color=cnames.values()[users.index(user_id)])
                        plt.plot([pt[0],pt[0]],[pt[1],pt[1]+pt[3]],color=cnames.values()[users.index(user_id)])
                        plt.plot([pt[0]+pt[2],pt[0]+pt[2]],[pt[1],pt[1]+pt[3]],color=cnames.values()[users.index(user_id)])
                        plt.plot([pt[0],pt[0]+pt[2]],[pt[1]+pt[3],pt[1]+pt[3]],color=cnames.values()[users.index(user_id)])
                    else:
                        print shape
                        assert False
            # e = numpy.e
            # circ = plt.Circle((5, 5), radius=2, color='g')
            # ax.add_patch(circ)
            #
            # X, Y = numpy.meshgrid(numpy.linspace(0, 5, 100), numpy.linspace(0, 5, 100))
            # F = X ** Y
            # G = Y ** X
            # # plt.contour(X, Y, (F - G), [0])
            plt.axis('scaled')
            print users
            plt.show()



    def __plot_cluster_results__(self):
        print "====="
        for subject_id in self.subjects:
            print subject_id

            url,fname = self.__get_subject_url_and_fname__(subject_id)

            image_path = base_directory+"/Databases/images/"+fname

            if not(os.path.isfile(image_path)):
                urllib.urlretrieve(url, image_path)

            image_file = cbook.get_sample_data(image_path)
            image = plt.imread(image_file)
            fig, ax = plt.subplots()
            im = ax.imshow(image)

            for task in self.cluster_alg.clusterResults[subject_id]:
                # not really a task - just there to make things easier to understand
                if task == "param":
                    continue
                for shape in self.cluster_alg.clusterResults[subject_id][task]:
                    print shape
                    for cluster in self.cluster_alg.clusterResults[subject_id][task][shape]:
                        center = cluster["center"]
                        if shape == "line":
                            plt.plot([center[0],center[1]],[center[2],center[3]],color="blue")


            plt.show()

    def __store_results__(self):
        aggregate_results = {}

        for subject_id in self.subjects:
            # there have been requests for the aggregation to also contain the metadata
            select = "SELECT metadata from subjects where id="+str(subject_id)
            cur = self.postgres_session.cursor()
            cur.execute(select)
            metadata = cur.fetchone()

            aggregation = {"metadata":metadata,"param":"task_id"}

            if self.cluster_alg is not None:
                cluster_results = self.cluster_alg.clusterResults[subject_id]

                # check to see if an aggregation already exists, if so, use that created_at time
                old_aggregation = self.cassandra_session.execute("select created_at from aggregations where subject_id = 1" + str(subject_id) + " and workflow_id = " + str(self.workflow_id))
                if old_aggregation == []:
                    created_at = datetime.datetime.now()
                else:
                    created_at = old_aggregation.created_at

                for key in cluster_results:
                    aggregation[key] = deepcopy(cluster_results[key])

            print aggregation

            # now merge in the classification aggregation results
            if self.classification_alg is not None:
                classification_results = self.classification_alg.results[subject_id]
                for task_id in self.classification_alg.results[subject_id]:
                    if task_id == "type":
                        continue

                    if task_id not in aggregation:
                        aggregation[task_id] = deepcopy(classification_results[task_id])
                    else:
                        for tool_id in cluster_results[task_id]:
                            if tool_id not in aggregation[task_id]:
                                aggregation[task_id][tool_id] = deepcopy(classification_results[task_id][tool_id])
                            else:
                                # if we are here, there should be some clustering results as well - which we
                                # need to merge with and not simply overwrite
                                for question_id in cluster_results[task_id][tool_id]:
                                    aggregation[task_id][tool_id][question_id] = deepcopy(classification_results[task_id][tool_id][question_id])

            print aggregation
        return

        updated_at = datetime.datetime.now()

        # insert into cassandra
        self.cassandra_session.execute(
            """
            insert into aggregations (subject_id,workflow_id, aggregation, created_at, updated_at,metadata)
            values (%s,%s,%s,%s,%s,%s)
            """,
            (subject_id,self.workflow_id,json.dumps(aggregation),created_at,updated_at,json.dumps(metadata)))

        # now insert into postgres as well
        self.postgres_cursor.execute("INSERT INTO aggregations(workflow_id,subject_id,aggregation,created_at,updated_at) VALUES (%s,%s,%s,%s,%s);",
                                     (self.workflow_id,subject_id,json.dumps(aggregation),created_at,updated_at))

    def __sort_classifications__(self):
        classifications = {}
        for subject_id in self.subjects:
            t_classifications = {}
            non_logged_in_users = 0

            for record in self.cassandra_session.execute("select user_id,annotations from classifications where project_id = " + str(self.project_id) + " and subject_id = " + str(subject_id) + " and workflow_id = " + str(self.workflow_id)):
                # convert from string into json
                annotations = json.loads(record.annotations)
                user_id = record.user_id
                if user_id == -1:
                    non_logged_in_users += -1
                    user_id = non_logged_in_users

                for task in annotations:
                    task_id = task["task"]

                    # need to separate "simple" classifications and those based on markings
                    # if based on markings, we need to use the clustering results
                    if task_id in self.classification_tasks:
                        if task_id in self.marking_params_per_task:
                            assert self.cluster_alg is not None


                            # go through each marking the user made, and see if there is a corresponding
                            # classification, if so, find the appropriate cluster
                            for marking in task["value"]:
                                print marking
                                # may result in merging from different markings which use the same tool - could be problems if those markings
                                # have different classification questions associated with them
                                # again - see https://github.com/zooniverse/Panoptes-Front-End/issues/525
                                # for discussion
                                tool = marking["tool"]

                                # are there classification questions associated with this marking tool specifically
                                if tool not in self.classification_tasks[task_id]:
                                    # if not, continue
                                    continue

                                # extract the necessary params
                                mapped_marking = [marking[p] for p in self.marking_params_per_task[task_id][tool]]

                                # find which cluster this point belongs to
                                cluster_found = False
                                for cluster_index,cluster in enumerate(self.cluster_alg.clusterResults[subject_id][task_id]):
                                    if mapped_marking in cluster["points"]:
                                        cluster_found = True
                                        break
                                assert cluster_found

                                if task_id not in t_classifications:
                                    t_classifications[task_id] = []
                                # extend the classifications list for this task as necessary
                                if len(t_classifications[task_id]) <= cluster_index:
                                    diff = cluster_index - len(t_classifications[task_id])+1

                                    # extend the classifications for each question
                                    question_list = [[] for i in self.classification_tasks[task_id][tool]]
                                    t_classifications[task_id].extend([question_list for i in range(diff)])

                                # finally add the details to the correct cluster
                                # might be an empty list if the user didn't answer the question
                                for question_id,classification in enumerate(marking["details"]):
                                    t_classifications[task_id][cluster_index][question_id].append((user_id,classification["value"]))
                        # else we have a simple classification - makes things so much easier :)
                        else:
                            if task_id not in t_classifications:
                                t_classifications[task_id] = []

                            t_classifications[task_id].append((user_id,task["value"]))

            classifications[subject_id] = deepcopy(t_classifications)
        return classifications

    def __classify__(self):
        individual_classification = self.__sort_classifications__()

        self.classification_alg.__aggregate__(self.subjects,individual_classification)

        if self.aggregations is None:
            self.aggregations = self.classification_alg.results
        else:
            # we must merge clustering and classification results
            for subject_id in self.classification_alg.results:
                if subject_id not in self.aggregations:
                    self.aggregations[subject_id] = self.classification_alg.results[subject_id]
                else:
                    for task_id in self.classification_alg.results[subject_id]:
                        if task_id not in self.aggregations[subject_id]:
                            self.aggregations[subject_id][task_id] = self.classification_alg.results[subject_id][task_id]
                        else:
                            # todo - fill this in
                            assert False
        print self.aggregations

        # for task_id in self.classification_tasks:
        #     # if we have a simple type of classification
        #     if isinstance(self.classification_tasks[task_id],bool):
        #         self.classification_alg.__classify__(self.subjects,task_id)
        #     else:
        #         assert False
        #         self.classification_alg.__classify__(self.subjects,task_id)


        #     # self.votes[subject_id] = votes_per_subject
        #     print json.dumps(classifications,indent=4,sort_keys=True)
        # return
        # # now that we gone through and extracted the annotations for each subject, go through according
        # # to question,task and tool index
        # for task_id in self.votes[subject_ids[0]]:
        #     if isinstance(self.votes[subject_ids[0]][task_id],list):
        #         self.classification_alg.__classify__(subject_ids,task_id)
        #     else:
        #         for tool_id in self.votes[subject_ids[0]][task_id].keys():
        #             for question_id in self.votes[subject_ids[0]][task_id][tool_id].keys():
        #                 self.classification_alg.__classify__(subject_ids,task_id,tool_id,question_id)

    def __set_subjects__(self,subjects):
        self.subjects = subjects

    def __get_subjects__(self):
        rows = self.cassandra_session.execute("select subject_id from classifications where project_id = " + str(self.project_id))
        subjects = set([r.subject_id for r in rows])
        print subjects

    def __set_classification_alg__(self,alg):
        self.classification_alg = alg(self)
        assert isinstance(self.classification_alg,classification.Classification)

    def __get_num_clusters_(self,subject_id,task_id):
        return len(self.cluster_alg.clusterResults[subject_id][task_id])

    def __get_classifications__(self,subject_id,task_id,cluster_index=None,question_id=None):
        # either both of these variables are None or neither of them are
        assert (cluster_index is None) == (question_id is None)

        if cluster_index is None:
            return self.classifications[subject_id][task_id]
        else:
            return self.classifications[subject_id][task_id][cluster_index][question_id]



if __name__ == "__main__":
    project = PanoptesAPI("bar_lengths")
    # project.__migrate__()

    # project.__get_subjects__()
    # # # brooke.__get_markings__(3266)
    # #
    project.__set_subjects__([3355,3273])

    import agglomerative
    project.__set_clustering_alg__(agglomerative.Agglomerative)
    # # # a = agglomerative.Agglomerative(brooke)
    project.__cluster__()

    project.__set_classification_alg__(classification.MajorityVote)
    project.__classify__()
    # project.__store_results__()