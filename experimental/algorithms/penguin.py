#!/usr/bin/env python
__author__ = 'greg'
import sys
sys.path.append("/home/greg/github/reduction/engine")
sys.path.append("/home/ggdhines/PycharmProjects/reduction/engine")
import automatic_optics
import pymongo
import agglomerative
import aggregation_api
# import panoptes_ibcc
import cassandra
import json
from cassandra.concurrent import execute_concurrent
from cassandra.cluster import Cluster
import urllib2
import os
import math
# import yaml
import csv
import matplotlib.pyplot as plt
# import classification
import matplotlib.cbook as cbook

global_workflow_id = -1
global_task_id = 1
global_version = 1

class Penguins(aggregation_api.AggregationAPI):
    def __init__(self):
        aggregation_api.AggregationAPI.__init__(self,project = -1, environment="penguins",panoptes_connect=False)
        self.project_id = -1
        # connect to the mongo server
        client = pymongo.MongoClient()
        db = client['penguin_2015-06-01']
        self.classification_collection = db["penguin_classifications"]
        self.subject_collection = db["penguin_subjects"]

        self.gold_standard = False
        # self.raw_markings,self.raw_classifications = self.__load_classifcations__()

        # some stuff to pretend we are a Panoptes project
        classification_tasks = {global_task_id:{"shapes":["point"]}}
        marking_tasks = {global_task_id:["point","point","point","point"]}

        database = {}

        self.workflows = {global_workflow_id:(classification_tasks,marking_tasks)}
        self.versions = {global_workflow_id:global_version}

        # self.cluster_algs = {"point":agglomerative.Agglomerative("point")}
        # self.classification_alg = classification.VoteCount()#panoptes_ibcc.IBCC()

        # self.__cassandra_connect__()

        self.classification_table = "penguins_classifications"
        self.users_table = "penguins_users"
        self.subject_id_type = "text"

        # self.postgres_session = psycopg2.connect("dbname='zooniverse' user=greg")


        self.experts = ["caitlin.black"]

        # self.postgres_cursor.execute("create table aggregations (workflow_id int, subject_id text, aggregation jsonb, created_at timestamp, updated_at timestamp)")

        # roi stuff

        with open(aggregation_api.base_directory+"/github/Penguins/public/roi.tsv","rb") as roi_file:
            roi_file.readline()
            reader = csv.reader(roi_file,delimiter="\t")
            for l in reader:
                path = l[0]
                t = [r.split(",") for r in l[1:] if r != ""]
                self.roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]

        # which subjects were taken at which site
        self.subject_to_site = {}

        default_clustering_algs = {"point":automatic_optics.AutomaticOptics}
        reduction_algs = {}
        # self.__set_clustering_algs__(default_clustering_algs,reduction_algs)

    def __cassandra_annotations__(self,workflow_id,subject_set):
        """
        get the annotations from Cassandra
        :return:
        """
        assert isinstance(subject_set,list) or isinstance(subject_set,set)

        version = int(math.floor(float(self.versions[workflow_id])))

        # todo - do this better
        width = 2000
        height = 2000

        classification_tasks,marking_tasks = self.workflows[workflow_id]
        raw_classifications = {}
        raw_markings = {}

        if subject_set is None:
            subject_set = self.__load_subjects__(workflow_id)

        total = 0

        # do this in bite sized pieces to avoid overwhelming DB
        for s in self.__chunks__(subject_set,15):
            statements_and_params = []

            if self.ignore_versions:
                select_statement = self.cassandra_session.prepare("select user_id,annotations,workflow_version from "+self.classification_table+" where project_id = ? and subject_id = ? and workflow_id = ?")
            else:
                select_statement = self.cassandra_session.prepare("select user_id,annotations,workflow_version from "+self.classification_table+" where project_id = ? and subject_id = ? and workflow_id = ? and workflow_version = ?")

            for subject_id in s:
                if self.ignore_versions:
                    params = (int(self.project_id),subject_id,int(workflow_id))
                else:
                    params = (int(self.project_id),subject_id,int(workflow_id),version)
                statements_and_params.append((select_statement, params))
            results = execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=False)

            for subject_id,(success,record_list) in zip(s,results):
                if not success:
                    print record_list
                assert success


                # seem to have the occasional "retired" subject with no classifications, not sure
                # why this is possible but if it can happen, just make a note of the subject id and skip
                if record_list == []:
                    # print "warning :: subject " + str(subject_id) + " has no classifications"
                    continue

                for ii,record in enumerate(record_list):
                    if record.user_id not in self.experts:
                        yield subject_id,record.user_id,record.annotations

        raise StopIteration()

    def __cassandra_connect__(self):
        """
        connect to the AWS instance of Cassandra - try 10 times and raise an error
        :return:
        """
        for i in range(10):
            try:
                self.cluster = Cluster()
                self.cassandra_session = self.cluster.connect('zooniverse')
                return
            except cassandra.cluster.NoHostAvailable:
                pass

        assert False



    def __get_related_subjects__(self,workflow_id,subject_id):
        stmt = "select user_id from " + str(self.classification_table) + " where project_id = " + str(self.project_id) + " and subject_id = '" + str(subject_id) + "' and workflow_id = " + str(global_workflow_id) + " and workflow_version = " + str(global_version)
        users_records = self.cassandra_session.execute(stmt)
        users = [u.user_id for u in users_records]

        all_subjects = {}

        for u in users:
            if u in self.experts:
                continue

            stmt = "select subject_id from penguins_users where project_id = " + str(self.project_id) + " and workflow_id = " + str(global_workflow_id) + " and workflow_version = " + str(global_version) + " and user_id = '" + str(u) + "'"
            subject_records = self.cassandra_session.execute(stmt)
            for r in subject_records:
                if r.subject_id == subject_id:
                    continue

                if r.subject_id not in all_subjects:
                    all_subjects[r.subject_id] = 1
                else:
                    all_subjects[r.subject_id] += 1

        related_subjects_and_counts =  sorted(all_subjects.items(), key = lambda x:x[1],reverse=True)
        # add the starting subject_id to the start - in case of a tie in the count, there is no guarantee that first
        # X subjects will contain the subject id
        related_subjects = [subject_id]
        related_subjects.extend(list(zip(*related_subjects_and_counts)[0]))
        return related_subjects

    def __get_retired_subjects__(self,workflow_id,with_expert_classifications=False):
        # project_id, workflow_id, subject_id,annotations,user_id,user_ip,workflow_version
        stmt = "select subject_id from penguins_users where project_id = " + str(self.project_id) + " and workflow_id = " + str(global_workflow_id) + " and workflow_version = " + str(global_version)
        if with_expert_classifications:
            stmt += " and user_id = '" + str(self.experts[0]) + "'"

        subjects = self.cassandra_session.execute(stmt)

        # if we don't require expert classifications, there are going to be multiple people who have classified an image
        # so subjects will show up more than once => use a set to make sure of uniqueness
        subject_set = list(set([r.subject_id for r in subjects]))
        print len(subject_set)
        return subject_set[:1000]

    def __get_expert_annotations__(self,workflow_id,subject_id):
        # todo- for now just use one expert
        version = str(int(math.floor(float(self.versions[workflow_id]))))
        stmt = """select annotations from """+ str(self.classification_table)+""" where project_id = """ + str(self.project_id) + """ and subject_id = '""" + str(subject_id) + """' and workflow_id = """ + str(workflow_id) + """ and workflow_version = """+ version + """ and user_id = '""" + str(self.experts[0]) + "'"
        # print stmt
        expert_annotations = self.cassandra_session.execute(stmt)
        return expert_annotations
        # print expert_annotations


    # def __aggregate__(self,workflows=None,subject_set=None,gold_standard_clusters=([],[])):
    #     # if not gold standard
    #     if not self.gold_standard:
    #         aggregation_api.AggregationAPI.__aggregate__(self,workflows,subject_set,gold_standard_clusters)


    def __get_correct_points__(self,workflow_id,subject_id,task_id,shape):
        """
        determine which markings are correct - as far as the gold standard data is concerned
        :param workflow_id:
        :param subject_id:
        :param task_id:
        :param shape:
        :return:
        """
        postgres_cursor = self.postgres_session.cursor()

        # the expert's classifications
        # this is from cassandra
        stmt = "select annotations from penguins_classifications where project_id = " + str(self.project_id) + " and subject_id = '" + str(subject_id) + "' and workflow_id = " + str(global_workflow_id) + " and workflow_version = "+str(global_version) + " and user_id = '" + str(self.experts[0]) + "'"
        r = self.cassandra_session.execute(stmt)
        expert_annotations = json.loads(r[0].annotations)

        print "experts say"
        print expert_annotations

        # get the markings made by the experts
        gold_pts = []
        for ann in expert_annotations[0]["value"]:
            gold_pts.append(aggregation_api.point_mapping(ann,(5000,5000)))

        # get the user markings
        stmt = "select aggregation from aggregations where workflow_id = " + str(workflow_id) + " and subject_id = '" + str(subject_id) + "'"
        postgres_cursor.execute(stmt)

        # todo - this should already be a dict but doesn't seem to be - hmmmm :/
        agg = postgres_cursor.fetchone()
        if agg is None:
            return []

        if isinstance(agg[0],str):
            aggregations = json.loads(agg[0])
        else:
            aggregations = agg[0]

        assert isinstance(aggregations,dict)
        print "users say"
        print aggregations

        cluster_centers = []
        for cluster_index,cluster in aggregations[str(task_id)][shape + " clusters"].items():
            if cluster_index == "param":
                continue
            cluster_centers.append(cluster["center"])

        # the three things we will want to return
        correct_pts = []
        # missed_pts = []
        # false_positives = []

        # if there are no gold markings, technically everything is a false positive
        if gold_pts == []:
            return []

        # if there are no user markings, we have missed everything
        if cluster_centers == []:
            return []

        # we know that there are both gold standard points and user clusters - we need to match them up
        # user to gold - for a gold point X, what are the user points for which X is the closest gold point?
        users_to_gold = [[] for i in range(len(gold_pts))]

        # find which gold standard pts, the user cluster pts are closest to
        # this will tell us which gold points we have actually found
        for local_index, u_pt in enumerate(cluster_centers):
            # dist = [math.sqrt((float(pt["x"])-x)**2+(float(pt["y"])-y)**2) for g_pt in gold_pts]
            min_dist = float("inf")
            closest_gold_index = None

            # find the nearest gold point to the cluster center
            # doing this in a couple of lines so that things are simpler - need to allow
            # for an arbitrary number of dimensions
            for gold_index,g_pt in enumerate(gold_pts):
                dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip(u_pt,g_pt)]))

                if dist < min_dist:
                    min_dist = dist
                    closest_gold_index = gold_index

            if min_dist < 30:
                users_to_gold[closest_gold_index].append(local_index)

        # and now find out which user clusters are actually correct
        # that will be the user point which is closest to the gold point
        distances_l =[]
        for gold_index,g_pt in enumerate(gold_pts):
            min_dist = float("inf")
            closest_user_index = None

            for u_index in users_to_gold[gold_index]:
                assert isinstance(u_index,int)
                dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip(cluster_centers[u_index],g_pt)]))

                if dist < min_dist:
                    min_dist = dist
                    closest_user_index = u_index

            # if none then we haven't found this point
            if closest_user_index is not None:
                assert isinstance(closest_gold_index,int)
                u_pt = cluster_centers[closest_user_index]
                correct_pts.append(tuple(u_pt))
                # todo: probably remove for production - only really useful for papers
                # self.user_gold_distance[subject_id].append((u_pt,g_pt,min_dist))
                # distances_l.append(min_dist)

                # self.user_gold_mapping[(subject_id,tuple(u_pt))] = g_pt

        return correct_pts



    def __get_aggregated_subjects__(self,workflow_id):
        """
        return a list of subjects which have aggregation results
        :param workflow_id:
        :return:
        """
        stmt = "select subject_id from aggregations where workflow_id = " + str(workflow_id)
        postgres_cursor = self.postgres_session.cursor()
        postgres_cursor.execute(stmt)

        subjects = []

        for r in postgres_cursor.fetchall():
            subjects.append(r[0])

        return subjects

    def __get_subject_dimension__(self,subject_id):
        subject = self.subject_collection.find_one({"zooniverse_id":subject_id})
        dim = subject["metadata"]["original_size"]

        return dim

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

    def __get_expert_markings__(self,subject_id):
        return self.classification_collection.find_one({"subjects.zooniverse_id":subject_id,"user_name":self.experts[0]})

    def __in_roi__(self,subject_id,marking):
        """
        does the actual checking
        :param object_id:
        :param marking:
        :return:
        """
        site = self.subject_to_site[subject_id]
        if site is None:
            return True

        roi = self.roi_dict[site]

        x = float(marking["x"])
        y = float(marking["y"])


        X = []
        Y = []

        for segment_index in range(len(roi)-1):
            rX1,rY1 = roi[segment_index]
            X.append(rX1)
            Y.append(-rY1)
        # if subject_id == "APZ0002do1":
        #     plt.plot(x,-y,"o")
        #     plt.plot(X,Y)
        #
        # if subject_id == "APZ0002do1":
        #     plt.show()

        # plt.show()

        # find the line segment that "surrounds" x and see if y is above that line segment (remember that
        # images are flipped)
        for segment_index in range(len(roi)-1):
            if (roi[segment_index][0] <= x) and (roi[segment_index+1][0] >= x):
                rX1,rY1 = roi[segment_index]
                rX2,rY2 = roi[segment_index+1]

                # todo - check why such cases are happening
                if rX1 == rX2:
                    continue

                m = (rY2-rY1)/float(rX2-rX1)
                rY = m*(x-rX1)+rY1

                if y >= rY:
                    # we have found a valid marking
                    # create a special type of animal None that is used when the animal type is missing
                    # thus, the marking will count towards not being noise but will not be used when determining the type

                    return True
                else:
                    return False

        # probably shouldn't happen too often but if it does, assume that we are outside of the ROI
        return False

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

            if "user_name" in classification:
                user_name = classification["user_name"]
            else:
                user_name = classification["user_ip"]

            user_ip = classification["user_ip"]

            mapped_annotations = [{"task":global_task_id,"value":[]}]

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
                        mapped_annotations[0]["value"][global_workflow_id]["tool"] = all_tools.index(marking["value"])

                except (AttributeError,KeyError) as e:
                    # print e
                    pass

            if mapped_annotations == {}:
                print "skipping"
                continue
            statements_and_params.append((insert_statement,(-1,-1,zooniverse_id,json.dumps(mapped_annotations),user_name,user_ip,1)))
            statements_and_params2.append((user_insert,(-1,-1,zooniverse_id,user_name,user_ip,1)))

        execute_concurrent(self.cassandra_session, statements_and_params, raise_on_first_error=True)

        execute_concurrent(self.cassandra_session, statements_and_params2, raise_on_first_error=True)

    def __roi_check__(self,marking,subject_id):
        """
        check if the marking is the in roi
        :param marking:
        :param subject_id:
        :return:
        """
        if subject_id not in self.subject_to_site:
            try:
                path = self.subject_collection.find_one({"zooniverse_id":subject_id})["metadata"]["path"]
            except TypeError:
                print subject_id
                print self.subject_collection.find_one({"zooniverse_id":subject_id})
                raise
            assert isinstance(path,unicode)
            slash_index = path.index("/")
            underscore_index = path.index("_")
            site_name = path[slash_index+1:underscore_index]

            # hard code some name changes in
            if site_name == "BOOTa2012a":
                site_name = "PCHAa2013"
            elif site_name == "BOOTb2013a":
                site_name = "PCHb2013"
            elif site_name == "DANCa2012a":
                site_name = "DANCa2013"
            elif site_name == "MAIVb2012a":
                site_name = "MAIVb2013"
            elif site_name == "NEKOa2012a":
                site_name = "NEKOa2013"
            elif site_name == "PETEa2013a":
                site_name = "PETEa2013a"
            elif site_name == "PETEa2013b":
                site_name = "PETEa2013a"
            elif site_name == "PETEb2012b":
                site_name = "PETEb2013"
            elif site_name == "SIGNa2013a":
                site_name = "SIGNa2013"

            if not(site_name in self.roi_dict.keys()):
                self.subject_to_site[subject_id] = None
            else:
                self.subject_to_site[subject_id] = site_name

        return self.__in_roi__(subject_id,marking)

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
        for subject in self.subject_collection.find({"tutorial":{"$ne":True}}).limit(5000):
            subjects.append(subject["zooniverse_id"])

        return subjects


class SubjectGenerator:
    def __init__(self,project):
        assert isinstance(project,aggregation_api.AggregationAPI)
        self.project = project

    def __iter__(self):
        subject_ids = []
        for subject in self.project.__get_retired_subjects__(1,False):
            subject_ids.append(subject)

            if len(subject_ids) == 5000:
                yield subject_ids
                subject_ids = []

        yield  subject_ids
        raise StopIteration

if __name__ == "__main__":
    project = Penguins()
    # project.__migrate__()
    # subjects = project.__get_subject_ids__()

    # print project.__get_retired_subjects__(1,True)

    for subject_id in project.__get_retired_subjects__(1,True)[:1]:
        fname = project.__image_setup__(subject_id)

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        image_file = cbook.get_sample_data(fname)
        image = plt.imread(image_file)
        # fig, ax = plt.subplots()
        im = axes.imshow(image)

        clusters = project.__aggregate__(workflows=[global_workflow_id],subject_set=[subject_id],store_values=False)[subject_id][1]["point clusters"]

        for cluster_index in clusters:
            if cluster_index not in ["param","all_users"]:
                center = clusters[cluster_index]["center"]
                plt.plot(center[0],center[1],"o",color="blue")
            # print clusters[cluster_index]

        for marking in project.__get_expert_markings__(subject_id)["annotations"][1]["value"].values():
            plt.plot(float(marking["x"]),float(marking["y"]),"o",color="green")

        # plt.ylim((520,360))
        plt.show()