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
    def __init__(self):
        # right now, each time we read in the classifications/annotations for a given subject, we are reading in all
        # of them that have ever been created, even if they were processed during a previous running of this program
        # so we shouldn't try to merge with previous results - would result in definite double counting
        # so do a hard reset with the aggregations
        #create a lock

        # do we have a lock from a previous - failed - run of this program (base or iterative)
        # doesn't really affect the base run but the iterative will need to start from scratch
        if os.path.isfile("/tmp/panoptes.lock"):
            self.fail = True
        else:
            self.fail = False
            f = open("/tmp/panoptes.lock","w")
            f.close()

        self.aggregations = []

        # try load previously read in metadata - if we can't we'll just have to download it again
        # not the end of the world
        try:
            self.metadata = pickle.load(open("/tmp/metadata.pickle","rb"))
        except IOError:
            self.metadata = []

        # # technically we could reuse the aggregation results to find the classification count but we want to store
        # # the counts from one running of this program to an other - so best is to use two different variables
        # # to store these values
        # # if we are doing a complete update, start from scratch
        # if update_type == "complete":
        #     self.classification_count = []
        # else:
        #     # loading the classification - if they don't exist, default to empty dicts
        #     # classification_count is useful for helping to determine which subjects we actually want to update
        #     try:
        #         self.classification_count = pickle.load(open("/tmp/classification_count.pickle","rb"))
        #     except IOError:
        #         self.classification_count = []
        self.threshold_date = datetime.datetime(2015,3,16,17,0,0)
        self.current_timestamp = self.threshold_date

    def __cleanup__(self):
        #pickle.dump(self.classification_count,open("/tmp/classification_count.pickle","wb"))
        pickle.dump(self.metadata,open("/tmp/metadata.pickle","wb"))
        pickle.dump(self.aggregations,open("/tmp/aggregations.pickle","wb"))

        pickle.dump(self.aggregations,open("/tmp/aggregations.pickle","wb"))
        pickle.dump(self.current_timestamp,open("/tmp/timestamp.pickle","wb"))
        #x = raw_input('What is your favourite colour?')
        os.remove("/tmp/panoptes.lock")

    def __get_timestamp__(self):
        return self.current_timestamp

    def __set_timestamp__(self,timestamp):
        self.current_timestamp = timestamp

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
            #print >> sys.stderr, "badly formed annotation - should skip"
            raise

    def __expand__(self,subject_id):
        while len(self.aggregations) <= subject_id:
            self.aggregations.append(None)
            #self.classification_count.append(0)
            self.metadata.append(None)

    def __update_subject__(self,subject_id,accumulated_scores,metadata):
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
        # the following four lines of code are taken (and slightly adapted from):
        # http://astro-wise.org/awesoft/awehome/AWBASE/common/math/statistics.py
        # calculate the weighted average - same as the average if we hadn't compressed all of the scores into
        # one accumulated value
        # score_sum = reduce(operator.add, accumulated_scores)
        # mean = reduce(operator.add, [weight*x for x, weight in zip([-1,1,3], accumulated_scores)])/score_sum

        #stdv  = reduce(operator.add, [weight*(x-mean)**2 for x, weight in zip([-1,1,3], accumulated_scores)])
        #stdev = math.sqrt(stdv/score_sum)

        values = accumulated_scores
        mean = (values[0]*-1 + values[1]*1 + values[2]*3)/float(sum(values))
        # note this gives a biased estimate
        stdev = math.sqrt(((-1-mean)**2*values[0] + (1-mean)**2*values[1] + (3-mean)**2*values[1])/float(sum(values)))

        # save the resulting aggregation - this is the first and only time these values are actually associated
        # with the particular subject id
        # extend the list if we need to
        self.__expand__(subject_id)

        #assert len(self.aggregations) == len(self.classification_count)
        self.aggregations[subject_id] = {"mean":mean,"std":stdev,"count":accumulated_scores}

        # and update the classification count
        #self.classification_count[subject_id] = sum(accumulated_scores)

        # only update metadata if it is not none
        self.__update_metadata__(subject_id,metadata)

    def __update_metadata__(self,subject_id,metadata):
        self.__expand__(subject_id)
        if metadata is not None:
            if isinstance(metadata,str):
                self.metadata[subject_id] = json.loads(metadata)
            else:
                self.metadata[subject_id] = metadata

    def __have_metadata__(self,subject_id):
        if len(self.metadata) <= subject_id:
            return False
        elif self.metadata[subject_id] is None:
            return False
        else:
            return True

    def __list_aggregations__(self,subjects_ids=None):
        """
        generate a list of subject ids and their corresponding aggregations in JSON form
        :param subjects_ids:
        :return:
        """
        # now seems a good a time as any to save the classification_count and scores
        if subjects_ids is None:
            subjects_ids = range(len(self.aggregations))

        print "the number of subjects is " + str(len(subjects_ids))

        for subject_id in subjects_ids:
            if self.aggregations[subject_id] is not None:
                yield subject_id,self.aggregations[subject_id]

    def __init__accumulator__(self,subject_id = None):
        return [0,0,0]

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

        try:
            accumulator[self.__score_index__(annotation)] += 1
        except IndexError:
            pass

        return accumulator

    def __aggregations_to_string__(self):
        """
        convert the aggregation into string format - useful for when you want to print the aggregations out to
        a csv file
        """
        # start by getting all aggregations that have at least 5 classifications
        subjects_to_print = [subject_id for subject_id,agg in enumerate(self.aggregations) if (agg is not None) and (sum(agg["count"]) >= 5)]
        # now sort these aggregations
        subjects_to_print.sort(key = lambda x: self.aggregations[x]["mean"],reverse = True)

        results = ""
        for subject_id in subjects_to_print:

            agg = self.aggregations[subject_id]
            # add the metadata first
            metadata = self.metadata[subject_id]
            print subject_id,agg
            try:
                if metadata is None:
                    # should never happen but just in case
                    results += str(subject_id) + ",NA,NA,NA,NA"
                else:
                    for property in ["candidateID","RA","DEC","mag","mjd"]:
                        try:
                            results += str(metadata[property]) + "," #+ str(metadata["RA"]) + "," + str(metadata["DEC"]) + "," + str(metadata["mag"]) + "," + str(metadata["mjd"])
                        except KeyError:
                            print >> sys.stderr, "missing property: " + property
                            if property == "candidateID":
                                results += str(subject_id) + ","
                            else:
                                results += "NA,"
            except TypeError:
                print metadata
                raise
            results += str(agg["mean"]) + "," + str(agg["std"]) + "," + str(agg["count"][0]) + "," + str(agg["count"][1]) + ","+ str(agg["count"][2]) + "\n"

        return results


class PanoptesAPI:
    def __init__(self,http_update=False): #Supernovae
        # first find out which environment we are working with
        self.environment = os.getenv('ENVIRONMENT', "staging")

        # next connect to the Panoptes http API

        # get my userID and password
        # purely for testing, if this file does not exist, try opening on Greg's computer
        try:
            panoptes_file = open("config/aggregation.yml","rb")
        except IOError:
            panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")
        api_details = yaml.load(panoptes_file)

        userid = api_details[self.environment]["name"]
        password = api_details[self.environment]["password"]
        host = api_details[self.environment]["host"] #"https://panoptes-staging.zooniverse.org/"
        owner = api_details[self.environment]["owner"] #"brian-testing" or zooniverse
        project_name = api_details[self.environment]["project_name"]
        app_client_id = api_details[self.environment]["app_client_id"]

        # get the token necessary to connect with panoptes
        self.http_api = panoptesPythonAPI.PanoptesAPI(host,userid,password,app_client_id)
        #panoptesPythonAPI.get_bearer_token(userid,password)

        # get the project id for Supernovae and the workflow version
        self.project_id = self.http_api.get_project_id(project_name,owner=owner) #or zooniverse
        self.workflow_version = self.http_api.get_workflow_version(self.project_id)
        self.workflow_id = self.http_api.get_workflow_id(self.project_id)

        # print "project id  is " + str(project_id)
        # print "workflow id is " + str(workflow_id)

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
        try:
            details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
            self.conn = psycopg2.connect(details)
        except:
            print "I am unable to connect to the database"
            raise

        self.aggregator = Aggregation()

        # assert update_type in ["partial","complete"]
        # self.update_type = update_type

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

        self.__get_all_metadata__()

    def __cleanup__(self):
        self.aggregator.__cleanup__()

    def __update__(self):

        num_updated = self.__update_aggregations__()

        # write out the results to an s3 bucket - file is a csv file labelled with day, hour and minute
        if self.S3_conn is not None:
            result_bucket = self.S3_conn.get_bucket("zooniverse-aggregation")
            k = Key(result_bucket)
            t = datetime.datetime.now()
            fname = str(t.day) + "_"  + str(t.hour) + "_" + str(t.minute)
            k.key = "Stargazing/"+self.environment+"/"+fname+".csv"
            csv_contents = "candidateID,RA,DEC,mag,mjd,mean,stdev,count0,count1,count2\n"
            csv_contents += self.aggregator.__aggregations_to_string__()
            k.set_contents_from_string(csv_contents)
        else:
            print >> sys.stderr, "not able to connect to S3 - did not write any results to S3"

        # if we also want to use the http interface to put the results back onto Panoptes - seems to be running
        # rather slowly right now, so use with caution
        # if self.http_update is True:
        #     self.__http_score__update__(subjects_to_update)
        return num_updated

    def __get_all_metadata__(self):
        # check to see if the metadata file exists in the first place, if so, don't bother trying to download it
        if not os.path.isfile("/tmp/metadata.pickle"):
            select = "SELECT id,metadata from subjects"
            cur = self.conn.cursor()
            cur.execute(select)
            rows = cur.fetchall()
            for subject_id,metadata in rows:
                print subject_id
                self.aggregator.__update_metadata__(subject_id,metadata)


    def __get_metadata__(self,subject_id):
        # do we need to get the metadata for this subject?
        if not self.aggregator.__have_metadata__(subject_id):
            select = "SELECT metadata from subjects where id = " + str(subject_id)
            cur2 = self.conn.cursor()
            cur2.execute(select)
            metadata = cur2.fetchone()[0]

        else:
            # none means there will be no update
            metadata = None

        return metadata

    def __update_aggregations__(self):
        """
        update
        :param additional_conditions: if you want to restrict the updates to certain subjects
        :return:
        """


        select = "SELECT subject_ids,annotations,created_at from classifications where project_id="+str(self.project_id)+" and workflow_id=" + str(self.workflow_id) + " ORDER BY subject_ids"
        cur = self.conn.cursor()
        cur.execute(select)

        current_subject_id = None
        annotation_accumulator = self.aggregator.__init__accumulator__()

        for count,(subject_ids,annotations,time_stamp) in enumerate(cur.fetchall()):

            #print count, subject_ids
            # have we moved on to a new subject?
            if subject_ids[0] != current_subject_id:
                # if this is not the first subject, aggregate the previous one
                if current_subject_id is not None:
                    # save the results of old/previous subject

                    # if by some chance all of the classifications we've read in have been discarded
                    # just skip it
                    if annotation_accumulator != [0,0,0]:
                        metadata = self.__get_metadata__(current_subject_id)
                        self.aggregator.__update_subject__(current_subject_id,annotation_accumulator,metadata)

                # reset and move on to the next subject
                current_subject_id = subject_ids[0]
                annotation_accumulator = self.aggregator.__init__accumulator__()


            annotation_accumulator = self.aggregator.__accumulate__(annotations,annotation_accumulator)

        # make sure we update the aggregation for the final subject we read in
        # on the very off chance that we haven't read in any classifications, double check
        if current_subject_id is not None:
            if annotation_accumulator != [0,0,0]:
                metadata = self.__get_metadata__(current_subject_id)
                self.aggregator.__update_subject__(current_subject_id,annotation_accumulator,metadata)

        return count

    # def __update_scores__(self,additional_conditions = ""):
    #     select = "SELECT subject_ids,annotations from classifications where project_id="+str(self.project_id)+" and workflow_id=" + str(self.workflow_id) + additional_conditions
    #     cur = self.conn.cursor()
    #     cur.execute(select)
    #     # cur.execute("SELECT annotations from classifications where project_id="+str(self.project_id)+" and workflow_id=" + str(self.workflow_id) + " and subject_ids = ARRAY["+str(subject_id)+"]")
    #     rows = cur.fetchall()
    #
    #     for subject_ids,annotations in rows:
    #         subject_id = subject_ids[0]
    #         self.aggregator.__update_score__(subject_id,annotations)
    #
    #     # makes other functions easier if we just return None
    #     return None

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

        for subject_id,aggregation in self.aggregator.__list_aggregations__(subjects_to_update):
            # try creating the aggregation
            status,explanation = panoptesPythonAPI.create_aggregation(self.workflow_id,subject_id,self.token,aggregation)

            # if we had a problem, try updating the aggregation
            if status == 400:
                aggregation_id,etag = panoptesPythonAPI.find_aggregation_etag(self.workflow_id,subject_id,self.token)
                panoptesPythonAPI.update_aggregation(self.workflow_id,self.workflow_version,subject_id,aggregation_id,self.token,aggregation,etag)

    def __find_classification_count__(self):
        """
        finds the classification count for every subject in the current project/workflow
        does not actually update the classification_count dict - we can use this function to determine
        which subjects have had enough additional classifications so that we show rerun the aggregation
        those subjects should have their classification count updated - otherwise, don't change the count
        :return:
        """
        new_classification_count = []
        ids = []

        cur = self.conn.cursor()
        #cur.execute("Select * from subject_sets where ")

        # keeping the below commands in as a reminder of how to do the relevant SQL commands
        # cur.execute("SELECT subject_id from set_member_subjects where subject_set_id=4 OR subject_set_id=3")
        # cur.execute("SELECT expert_set from subject_sets where subject_sets.project_id="+str(project_id)+" and subject_sets.workflow_id=" + str(workflow_id))


        cur.execute("SELECT subject_id,classification_count from set_member_subjects inner join subject_sets on set_member_subjects.subject_set_id=subject_sets.id where (subject_sets.expert_set = FALSE or subject_sets.expert_set IS NULL) and subject_sets.project_id="+str(self.project_id)+" and subject_sets.workflow_id=" + str(self.workflow_id) +" ORDER BY subject_id")
        rows = cur.fetchall()
        for subject_id,count in rows:
            if count > 0:
                new_classification_count.append(count)
                ids.append(subject_id)

        return ids,new_classification_count

    def __heuristic_update(self):
        ids,new_classification_count = self.__find_classification_count__()
        subjects_to_update = self.aggregator.__subjects_to_update__(ids,new_classification_count)

        for subject_id in subjects_to_update:
            self.__update_aggregations(additional_conditions=" and subject_ids = ARRAY["+str(subject_id)+"]")

        return subjects_to_update



if __name__ == "__main__":
    update = "complete"
    http_update = True
    start = datetime.datetime.now()
    try:
        opts, args = getopt.getopt(sys.argv[1:],"u:m:",["update=",])
    except getopt.GetoptError:
        print "postgres_aggregation -u <COMPLETE or PARTIAL update> -m <http update method TRUE or FALSE>"
        sys.exit(2)

    for opt,arg in opts:
        # are we doing a partial or complete update?
        if opt in ["-u", "-update"]:
            update = arg.lower()
            assert update in ["complete", "partial"]
        elif opt in ["-m", "-method"]:
            http_update = arg.lower()
            # convert from string into boolean
            http_update = (http_update[0] == "t")

    # hard code this for now
    http_update = False
    stargazing = PanoptesAPI(http_update=http_update)
    num_updated = stargazing.__update__()

    # cleanup makes sure that we are dumping the aggregation results back to disk
    stargazing.__cleanup__()

    end = datetime.datetime.now()
    print "updated " + str(num_updated) + " in " + str(end - start)