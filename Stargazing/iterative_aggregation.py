#!/usr/bin/env python
__author__ = 'greg'
import math
import panoptesPythonAPI
import yaml
import psycopg2
import os
import sys
import cPickle as pickle
import getopt
import boto
from boto.s3.key import Key
import datetime
import numpy as np
import operator
import json


# for Greg running on either office/home - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

class AnnotationException(Exception):
    def __init__(self, value):
        self.annotation, self.index, self.task = value

    def __str__(self):
        return "With annotation: " + str(self.annotation) + " at index " + str(self.index) + " did not find task: " + str(self.task)


class Aggregation:
    def __init__(self,complete_update,partial_print):
        """
        :param complete_update: do we just read in stuff since the last run or everything from the beginning
        :param partial_print: do we just print out stuff that has been updated or everything>
        :return:
        """
        self.complete_update = complete_update
        self.partial_print = partial_print

    #def __enter__(self):
        # are we forcing a complete restart of the analysis - probably not going to happen very often (or at least
        # it shouldn't have to). Basically the same thing as if the lock file was left in place but in that case
        # we want to print out an error message

        threshold_date = datetime.datetime(2015,3,16,17,0,0)

        if self.complete_update:
            self.aggregations = []
            self.current_timestamp = threshold_date

            # create lock - if this code fails for whatever reason, we don't have any future iterations failing
            f = open("/tmp/panoptes.lock","w")
            f.close()
            assert os.path.isfile("/tmp/panoptes.lock")


        # check to see if the lock exists from a previous run which would indicate that that run failed
        # which could result in data corruption - since we don't know where the program crashed
        # so to play it safe, we will start over
        elif os.path.isfile("/tmp/panoptes.lock"):
            print >> sys.stderr, "lock file exists from previous run - starting over from scratch to avoid any data corruption"
            self.aggregations = []
            self.current_timestamp = threshold_date
        else:
            # create lock - if this code fails for whatever reason, we don't have any future iterations failing
            f = open("/tmp/panoptes.lock","w")
            f.close()
            assert os.path.isfile("/tmp/panoptes.lock")

            # we are going to try to load previous stuff - if this fails (maybe because this is the first
            # time we have run this program or we just had a reboot etc.), init with empty values
            try:
                self.aggregations = pickle.load(open("/tmp/aggregations.pickle","rb"))
                self.current_timestamp = pickle.load(open("/tmp/timestamp.pickle","rb"))
            except IOError:
                self.classification_count = []
                self.aggregations = []
                # makes it slightly easier to have an actual date in this variable
                # the value doesn't matter as long as it is before any classifications
                self.current_timestamp = threshold_date

        self.updated_aggregations = []

    def __get_timestamp__(self):
        return self.current_timestamp

    def __set_timestamp__(self,timestamp):
        self.current_timestamp = timestamp

    def __clean__(self):#, type, value, traceback):
        pickle.dump(self.aggregations,open("/tmp/aggregations.pickle","wb"))
        pickle.dump(self.current_timestamp,open("/tmp/timestamp.pickle","wb"))
        #x = raw_input('What is your favourite colour?')
        os.remove("/tmp/panoptes.lock")

    def __score_index__(self,annotations):
        """calculate the score associated with a given classification according to the algorithm
        in the paper Galaxy Zoo Supernovae
        an example of the json format used is
        [{u'task': u'centered_in_crosshairs', u'value': 1}, {u'task': u'subtracted', u'value': 1}, {u'task': u'circular', u'value': 1}, {u'task': u'centered_in_host', u'value': 0}]
        """
        try:
            if annotations[0]["task"] != "centered_in_crosshairs":
                raise AnnotationException((annotations,0,"centered_in_crosshairs"))
            if annotations[0]["value"] == 1:
                return 0  #-1

            if annotations[1]["task"] != "subtracted":
                raise AnnotationException((annotations,1,"subtracted"))
            if annotations[1]["value"] == 1:
                return 0  #-1


            if annotations[2]["task"] != "circular":
                raise AnnotationException((annotations,2,"circular"))
            if annotations[2]["value"] == 1:
                return 0  #-1

            if annotations[3]["task"] != "centered_in_host":
                raise AnnotationException((annotations,3,"centered_in_host"))
            if annotations[3]["value"] == 1:
                return 2  #3
            else:
                return 1  #1
        except IndexError:
            print >> sys.stderr, "Index error with annotations - did someone change yes/no again?"
            raise

    def __update_subject__(self,subject_id,accumulated_scores):
        """
        set the aggregation result for the given subject_id using the accumulated scores
        for now, assume that the accumulated scores are based on ALL classifications, including ones
        we have read in before hand, so we completely overwrite all pre-existing results. In the future we
        could merge the results - this means we only have to read in new classifications, but at the same time
        we have to sure that we don't read in any old classifications
        :param subject_id:
        :param accumulated_scores:
        :return:
        """
        assert isinstance(subject_id,int)
        #print "updating " + str(subject_id)
        # the following four lines of code are taken (and slightly adapted from):
        # http://astro-wise.org/awesoft/awehome/AWBASE/common/math/statistics.py
        # calculate the weighted average - same as the average if we hadn't compressed all of the scores into
        # one accumulated value
        score_sum = reduce(operator.add, accumulated_scores)
        mean = reduce(operator.add, [weight*x for x, weight in zip([-1,1,3], accumulated_scores)])/score_sum

        stdv  = reduce(operator.add, [weight*(x-mean)**2 for x, weight in zip([-1,1,3], accumulated_scores)])
        stdev = math.sqrt(stdv/score_sum)

        # save the resulting aggregation - this is the first and only time these values are actually associated
        # with the particular subject id
        # extend the list if we need to
        while len(self.aggregations) <= subject_id:
            self.aggregations.append(None)

        self.aggregations[subject_id] = {"mean":mean,"std":stdev,"count":accumulated_scores}

        # and update the classification count
        self.updated_aggregations.append(subject_id)


    def __init__accumulator__(self, subject_id):
        if subject_id is None:
            return [0,0,0]
        # we have not encountered this subject before
        # note that subjects are 1 indexed but the lists are 0-indexed
        elif len(self.aggregations) <= subject_id:
            return [0,0,0]
        elif self.aggregations[subject_id] is None:
            return [0,0,0]
        else:
            return self.aggregations[subject_id]["count"][:]

    def __accumulate__(self,annotation,accumulator):
        """
        :param annotation - the next annotation to "merge into" the set of existing annotations
         for future work - the merge into could be just appending to a list but here we should be able to keep
         the memory usage low
        :param accumulator - the merged results from the previous annotations
        :return:
        """
        # if the accumulator was read in directly from the sql file - we will need to convert it into json format
        if isinstance(annotation,str):
            annotation = json.loads(annotation)

        accumulator[self.__score_index__(annotation)] += 1

        return accumulator

    def __aggregations_to_string__(self):
        """
        convert the aggregation into string format - useful for when you want to print the aggregations out to
        a csv file
        """
        results = ""
        for subject_id,agg in enumerate(self.aggregations):
            if agg is not None:
                results += str(subject_id) + "," + str(agg["mean"]) + "," + str(agg["std"]) + "," + str(agg["count"][0]) + "," + str(agg["count"][1]) + ","+ str(agg["count"][2]) + "\n"

        return results


class PanoptesAPI:
    def __init__(self,aggregator,project_name,update_type="complete",http_update=True): #Supernovae
        # get my userID and password
        # purely for testing, if this file does not exist, try opening on Greg's computer
        try:
            panoptes_file = open("config/aggregation.yml","rb")
        except IOError:
            panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")
        login_details = yaml.load(panoptes_file)
        userid = login_details["name"]
        password = login_details["password"]

        # get the token necessary to connect with panoptes
        self.token = panoptesPythonAPI.get_bearer_token(userid,password)

        # get the project id for Supernovae and the workflow version
        self.project_id = panoptesPythonAPI.get_project_id(project_name,self.token)
        self.workflow_version = panoptesPythonAPI.get_workflow_version(self.project_id,self.token)
        self.workflow_id = panoptesPythonAPI.get_workflow_id(self.project_id,self.token)

        # print "project id  is " + str(project_id)
        # print "workflow id is " + str(workflow_id)

        # now load in the details for accessing the database
        try:
            database_file = open("config/database.yml")
        except IOError:
            database_file = open(base_directory+"/Databases/database.yml")
        database_details = yaml.load(database_file)

        #environment = "staging"
        self.environment = os.getenv('ENVIRONMENT', "staging")
        database = database_details[self.environment]["database"]
        username = database_details[self.environment]["username"]
        password = database_details[self.environment]["password"]
        host = database_details[self.environment]["host"]

        # try connecting to the db
        try:
            details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
            self.conn = psycopg2.connect(details)
        except:
            print "I am unable to connect to the database"
            raise

        self.aggregator = aggregator #Aggregation(update_type)

        assert update_type in ["partial","complete"]
        self.update_type = update_type

        assert http_update in [True,False]
        self.http_update = http_update

        self.S3_conn = None
        try:
            # for dumping results to s3
            AWS_ACCESS_KEY_ID  = os.getenv('AWS_ACCESS_KEY_ID', None)
            AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)

            # if both are None, we should already be in aws
            self.S3_conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
        except boto.exception.NoAuthHandlerFound:
            print >> sys.stderr, "not able to connect to S3 - will try to keep going but this will probably give you errors later"

    def __update__(self):
        # actually update the aggregations
        self.__update_aggregations()

        # write out the results to an s3 bucket - file is a csv file labelled with day, hour and minute
        if self.S3_conn is not None:
            sys.stdout.flush()
            result_bucket = self.S3_conn.get_bucket("zooniverse-aggregation")
            k = Key(result_bucket)
            t = datetime.datetime.now()
            fname = str(t.day) + "_"  + str(t.hour) + "_" + str(t.minute)
            k.key = "Stargazing/"+self.environment+"/"+fname+".csv"
            csv_contents = "subject_id,mean,std,count0,count1,count2\n"
            csv_contents += self.aggregator.__aggregations_to_string__()
            k.set_contents_from_string(csv_contents)
        else:
            print >> sys.stderr, "not able to connect to S3 - did not write any results to S3"

        # if we also want to use the http interface to put the results back onto Panoptes - seems to be running
        # rather slowly right now, so use with caution
        if self.http_update is True:
            self.__http_score__update__()

    def __update_aggregations(self):
        """
        update
        :param additional_conditions: if you want to restrict the updates to certain subjects
        :return:
        """
        current_timestamp = self.aggregator.__get_timestamp__()

        select = "SELECT subject_ids,annotations,created_at from classifications where project_id="+str(self.project_id)+" and workflow_id=" + str(self.workflow_id) + " and created_at>\'" + str(current_timestamp) + "\' ORDER BY subject_ids"

        cur = self.conn.cursor()
        cur.execute(select)

        current_subject_ids = None
        annotation_accumulator = self.aggregator.__init__accumulator__(None)

        #max_timestamp = datetime.datetime(2000,1,1,1,1)

        for subject_ids,annotations, date in cur.fetchall():
            print date
            # update the maximum time stamp if necessary
            current_timestamp = max(current_timestamp,date)
            # have we moved on to a new subject?
            if subject_ids != current_subject_ids:
                # if this is not the first subject, aggregate the previous one
                if current_subject_ids is not None:
                    # save the results of old/previous subject
                    self.aggregator.__update_subject__(current_subject_ids[0],annotation_accumulator)
                # reset and move on to the next subject
                current_subject_ids = subject_ids
                annotation_accumulator = self.aggregator.__init__accumulator__(subject_ids[0])

            annotation_accumulator = self.aggregator.__accumulate__(annotations,annotation_accumulator)

        # make sure we update the aggregation for the final subject we read in
        # on the very off chance that we haven't read in any classifications, double check
        if current_subject_ids is not None:
            self.aggregator.__update_subject__(current_subject_ids[0],annotation_accumulator)

        self.aggregator.__set_timestamp__(current_timestamp)

    def __http_score__update__(self,subjects_to_update=None):
        """
        after we have updated the scores, return them to panoptes using the http api
        the alternative might be to upload to s3 or email someone the results
        :param subjects_to_update: if there are a certain select number of subjects you want to update
        so specifically, you used a heuristic to select which subjects to update and only want to send
        those back. Basically if you selected heuristic for update and after sending everything back, that is
        kinda silly - hence the assert
        :return:
        """
        assert (self.update_type == "complete") or (subjects_to_update is not None)

        for subject_id,aggregation in self.aggregator.__list_aggregations__():
            # try creating the aggregation
            status,explanation = panoptesPythonAPI.create_aggregation(self.workflow_id,subject_id,self.token,aggregation)

            # if we had a problem, try updating the aggregation
            if status == 400:
                aggregation_id,etag = panoptesPythonAPI.find_aggregation_etag(self.workflow_id,subject_id,self.token)
                panoptesPythonAPI.update_aggregation(self.workflow_id,self.workflow_version,subject_id,aggregation_id,self.token,aggregation,etag)





if __name__ == "__main__":
    complete_update = True
    http_update = False
    partial_print = False

    try:
        opts, args = getopt.getopt(sys.argv[1:],"u:m:",["update=",])
    except getopt.GetoptError:
        print "postgres_aggregation -u <COMPLETE or ITERATIVE update> -m <http update method TRUE or FALSE> -p <COMPLETE or PARTIAL print out>"
        sys.exit(2)

    for opt,arg in opts:
        # are we doing a partial or complete update?
        if opt in ["-u", "-update"]:
            complete_update = arg.lower()
            complete_update = (complete_update[0] == "c")
        elif opt in ["-m", "-method"]:
            http_update = arg.lower()
            # convert from string into boolean
            http_update = (http_update[0] == "t")
        elif opt in ["-p", "-print"]:
            partial_print = arg.lower()
            partial_print = (partial_print[0] == "p")

    aggregate =  Aggregation(complete_update,partial_print)
    stargazing = PanoptesAPI(aggregate,"Snapshot+Supernova",http_update=http_update)
    stargazing.__update__()

    aggregate.__clean__()


