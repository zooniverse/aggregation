#!/usr/bin/env python
from __future__ import print_function
from aggregation_api import AggregationAPI
from classification import Classification
import rollbar
import requests
import os
import pickle
import helper_functions
import yaml
import json
from blob_clustering import BlobClustering
import parser
import getopt
import sys
import folger
import annotate
import numpy as np
import boto3
import tarfile
from helper_functions import warning

__author__ = 'ggdhines'


def get_signed_url(time, bucket, obj):
    """
    from https://gist.github.com/richarvey/637cd595362760858496
    :param time:
    :param bucket:
    :param obj:
    :return:
    """
    s3 = boto3.resource('s3')

    url = s3.generate_url(
        time,
        'GET',
        bucket,
        obj,
        response_headers={
          'response-content-type': 'application/octet-stream'
        }
    )
    return url


class SubjectRetirement(Classification):
    def __init__(self,environment,param_dict):
        Classification.__init__(self,environment)
        assert isinstance(param_dict,dict)

        # to retire subjects, we need a connection to the host api, which hopefully is provided
        self.host_api = None
        self.project_id = None
        self.token = None
        self.workflow_id = None
        for key,value in param_dict.items():
            if key == "host":
                self.host_api = value
            elif key == "project_id":
                self.project_id = value
            elif key == "token":
                self.token = value
            elif key == "workflow_id":
                self.workflow_id = value

        self.num_retired = None
        self.non_blanks_retired = None

        self.to_retire = None

        assert (self.host_api is not None) and (self.project_id is not None) and (self.token is not None) and (self.workflow_id is not None)

    def __aggregate__(self,raw_classifications,workflow,aggregations):
        # start by looking for empty subjects

        self.to_retire = set()
        for subject_id in raw_classifications["T0"]:
            user_ids,is_subject_empty = zip(*raw_classifications["T0"][subject_id])
            if is_subject_empty != []:
                empty_count = sum([1 for i in is_subject_empty if i == True])
                if empty_count >= 3:
                    self.to_retire.add(subject_id)

        blank_retirement = len(self.to_retire)

        non_blanks = []

        # now look to see if everything has been transcribed
        for subject_id in raw_classifications["T3"]:
            user_ids,completely_transcribed = zip(*raw_classifications["T3"][subject_id])

            completely_count = sum([1 for i in completely_transcribed if i == True])
            if completely_count >= 3:
                self.to_retire.add(subject_id)
                non_blanks.append(subject_id)

            # # have at least 4/5 of the last 5 people said the subject has been completely transcribed?
            # recent_completely_transcribed = completely_transcribed[-5:]
            # if recent_completely_transcribed != []:
            #     complete_count = sum([1 for i in recent_completely_transcribed if i == True])/float(len(recent_completely_transcribed))
            #
            #     if (len(recent_completely_transcribed) == 5) and (complete_count >= 0.8):
            #         to_retire.add(subject_id)
        print(self.to_retire)
        # don't retire if we are in the development environment
        if self.to_retire != set():
            try:
                headers = {"Accept":"application/vnd.api+json; version=1","Content-Type": "application/json", "Authorization":"Bearer "+self.token}
                params = {"retired_subjects":list(self.to_retire)}
                # r = requests.post("https://panoptes.zooniverse.org/api/workflows/"+str(self.workflow_id)+"/links/retired_subjects",headers=headers,json=params)
                r = requests.post("https://panoptes.zooniverse.org/api/workflows/"+str(self.workflow_id)+"/retired_subjects",headers=headers,data=json.dumps(params))
                print(r)
                print(r.text)
                # rollbar.report_message("results from trying to retire subjects","info",extra_data=r.text)

            except TypeError as e:
                warning(e)
                rollbar.report_exc_info()
        if self.environment == "development":
            print("we would have retired " + str(len(self.to_retire)))
            print("with non-blanks " + str(len(self.to_retire)-blank_retirement))
            if not os.path.isfile("/home/ggdhines/"+str(self.project_id)+".retired"):
                pickle.dump(non_blanks,open("/home/ggdhines/"+str(self.project_id)+".retired","wb"))
            print(str(len(self.to_retire)-blank_retirement))

        self.num_retired = len(self.to_retire)
        self.non_blanks_retired = len(self.to_retire)-blank_retirement

        return aggregations


class TranscriptionAPI(AggregationAPI):
    def __init__(self,project_id,environment,end_date=None):
        AggregationAPI.__init__(self,project_id,environment,end_date=end_date)

        # just to stop me from using transcription on other projects
        assert int(project_id) in [245,376]

    def __aggregate__(self,gold_standard_clusters=([],[]),expert=None):
        """
        you can provide a list of clusters - hopefully examples of both true positives and false positives
        note this means you have already run the aggregation before and are just coming back with
        more info
        for now, only one expert - easily generalizable but just want to avoid the situation where
        multiple experts have marked the same subject - need to be a bit careful there
        :param workflows:
        :param subject_set:
        :param gold_standard_clusters:
        :return:
        """
        # start by migrating any new classifications (since previous run) from postgres into cassandra
        # this will also give us a list of the migrated subjects, which is the list of subjects we want to run
        # aggregation on (if a subject has no new classifications, why bother rerunning aggregation)
        # this is actually just for projects like annotate and folger where we run aggregation on subjects that
        # have not be retired. If we want subjects that have been specifically retired, we'll make a separate call
        # for that
        for workflow_id,version in self.versions.items():
            migrated_subjects = self.__migrate__(workflow_id,version)

            # the migrated_subject can contain classifications for subjects which are not yet retired
            # so if we want only retired subjects, make a special call
            # otherwise, use the migrated list of subjects
            if self.only_retired_subjects:
                subject_set = self.__get_newly_retired_subjects__(workflow_id)
            else:
                subject_set = list(migrated_subjects[workflow_id])

            if subject_set == []:
                print("skipping workflow " + str(workflow_id) + " due to an empty subject set")
                subject_set = None
                continue

            print("workflow id : " + str(workflow_id))
            print("aggregating " + str(len(subject_set)) + " subjects")

            # self.__describe__(workflow_id)
            classification_tasks,marking_tasks,survey_tasks = self.workflows[workflow_id]

            # set up the clustering algorithms for the shapes we actually use
            used_shapes = set()
            for shapes in marking_tasks.values():
                used_shapes = used_shapes.union(shapes)

            aggregations = {}

            # image_dimensions can be used by some clustering approaches - ie. for blob clustering
            # to give area as percentage of the total image area
            raw_classifications,raw_markings,raw_surveys,image_dimensions = self.__sort_annotations__(workflow_id,subject_set,expert)

            # do we have any marking tasks?
            if False:#marking_tasks != {}:
                aggregations = self.__cluster2__(used_shapes,raw_markings,image_dimensions,raw_classifications)
                # assert (clustering_aggregations != {}) and (clustering_aggregations is not None)

            # we ALWAYS have to do classifications - even if we only have marking tasks, we need to do
            # tool classification and existence classifications
            print("classifying")
            aggregations = self.classification_alg.__aggregate__(raw_classifications,self.workflows[workflow_id],aggregations)

            # finally, store the results
            self.__upsert_results__(workflow_id,aggregations)

    def __cluster2__(self,used_shapes,raw_markings,image_dimensions,raw_classifications):
        """
        run the clustering algorithm for a given workflow
        need to have already checked that the workflow requires clustering
        :param workflow_id:
        :return:
        """

        if raw_markings == {}:
            warning("warning - empty set of images")
            return {}

        # start by clustering text
        print("clustering text")
        cluster_aggregation = self.text_algorithm.__aggregate__(raw_markings,image_dimensions,raw_classifications)
        print("clustering images")
        image_aggregation = self.image_algorithm.__aggregate__(raw_markings,image_dimensions)

        cluster_aggregation = self.__merge_aggregations__(cluster_aggregation,image_aggregation)

        return cluster_aggregation

    def __setup__(self):
        AggregationAPI.__setup__(self)

        workflow_id = self.workflows.keys()[0]

        # set the classification algorithm which will retire the subjects
        classification_params = {"host":self.host_api,"project_id":self.project_id,"token":self.token,"workflow_id":workflow_id}
        self.__set_classification_alg__(SubjectRetirement,classification_params)

        self.instructions[workflow_id] = {}

        # set the function which will extract the relevant params for processing transcription annotations
        self.marking_params_per_shape["text"] = helper_functions.relevant_text_params

        # set up the text clustering algorithm
        # todo - this might not be necesary anymore
        additional_text_args = {"reduction":helper_functions.text_line_reduction}

        # load in the tag file if there is one
        api_details = yaml.load(open("/app/config/aggregation.yml","rb"))
        if "tags" in api_details[self.project_id]:
            additional_text_args["tags"] = api_details[self.project_id]["tags"]

        # now that we have the additional text arguments, convert text_algorithm from a class
        # to an actual instance
        if self.project_id == 245:
            self.text_algorithm = annotate.AnnotateClustering("text",self,additional_text_args)
        elif self.project_id == 376:
            self.text_algorithm = folger.FolgerClustering("text",self,additional_text_args)
        else:
            assert False

        self.image_algorithm = BlobClustering("image",None,{})

        self.only_retired_subjects = False
        self.only_recent_subjects = True

    def __cluster_output_with_colour__(self,workflow_id,ax,subject_id):
        """
        use colour to show where characters match and don't match between different transcriptions of
        the same text
        :param subject_id:
        :return:
        """
        selection_stmt = "SELECT aggregation FROM aggregations WHERE workflow_id = " + str(workflow_id) + " AND subject_id = " + str(subject_id)
        cursor = self.postgres_session.cursor()
        cursor.execute(selection_stmt)

        aggregated_text = cursor.fetchone()[0]["T2"]["text clusters"].values()
        assert isinstance(aggregated_text,list)
        # remove the list of all users
        aggregated_text = [a for a in aggregated_text if isinstance(a,dict)]

        # sort the text by y coordinates (should give the order in which the text is supposed to appear)
        aggregated_text.sort(key = lambda x:x["center"][2])

        for text in aggregated_text:
            ax.plot([text["center"][0],text["center"][1]],[text["center"][2],text["center"][3]],color="red")
            actual_text = text["center"][-1]
            atomic_text = self.cluster_algs["text"].__set_special_characters__(actual_text)[1]

            for c in atomic_text:
                if ord(c) == 27:
                    # no agreement was reached
                    print(chr(8) + unicode(u"\u2224"),)
                elif ord(c) == 28:
                    # the agreement was that nothing was here
                    # technically not a space but close enough
                    print(chr(8) + " ",)
                else:
                    print(chr(8) + c,)
            print()

    def __readin_tasks__(self,workflow_id):
        if self.project_id == 245:
            # marking_tasks = {"T2":["image"]}
            marking_tasks = {"T2":["text","image"]}
            # todo - where is T1?
            classification_tasks = {"T0":True,"T3" : True}

            return classification_tasks,marking_tasks,{}
        elif self.project_id == 376:
            marking_tasks = {"T2":["text"]}
            classification_tasks = {"T0":True,"T3":True}

            print(AggregationAPI.__readin_tasks__(self,workflow_id))

            return classification_tasks,marking_tasks,{}
        else:
            assert False

    def __restructure_json__(self):
        workflow_id = self.workflows.keys()[0]

        cur = self.postgres_session.cursor()

        stmt = "select subject_id,aggregation from aggregations where workflow_id = " + str(workflow_id)
        cur.execute(stmt)

        new_json = {}

        subjects_with_results = 0

        for ii,(subject_id,aggregation) in enumerate(cur.fetchall()):
            #
            if subject_id not in self.classification_alg.to_retire:
                continue

            try:
                clusters_by_line = {}

                for key,cluster in aggregation["T2"]["text clusters"].items():
                    if key == "all_users":
                        continue

                    index = cluster["set index"]
                    # text_y_coord.append((cluster["center"][2],cluster["center"][-1]))

                    if index not in clusters_by_line:
                        clusters_by_line[index] = [cluster]
                    else:
                        clusters_by_line[index].append(cluster)

                cluster_set_coordinates = {}

                for set_index,cluster_set in clusters_by_line.items():
                    # clusters are based on purely horizontal lines so we don't need to take the
                    # average or anything like that.
                    # todo - figure out what to do with vertical lines, probably keep them completely separate
                    cluster_set_coordinates[set_index] = cluster_set[0]["center"][2]

                sorted_sets = sorted(cluster_set_coordinates.items(), key = lambda x:x[1])

                text_to_read = []
                text_in_detail = []
                coordinates = []
                for set_index,_ in sorted_sets:
                    cluster_set = clusters_by_line[set_index]

                    # now on the (slightly off chance) that there are multiple clusters for this line, sort them
                    # by x coordinates
                    line = [(cluster["center"][0],cluster["center"][-1]) for cluster in cluster_set]
                    line.sort(key = lambda x:x[0])
                    _,text = zip(*line)

                    text = list(text)
                    # for combining the possible multiple clusters for this line into one
                    merged_line = ""
                    for t in text:
                        # think that storing in postgres converts from str to unicode
                        # for general display, we don't need ord(24) ie skipped characters
                        new_t = t.replace(chr(24),"")
                        merged_line += new_t

                    # we seem to occasionally get lines that are just skipped characters (i.e. the string
                    # if just chr(24)) - don't report these lines
                    if merged_line != "":
                        # is this the first line we've encountered for this subject?
                        if subject_id not in new_json:
                            new_json[subject_id] = {"text":[],"individual transcriptions":[], "accuracy":[], "coordinates" : []}

                            # add in the metadata
                            metadata = self.__get_subject_metadata__(subject_id)["subjects"][0]["metadata"]
                            new_json[subject_id]["metadata"] = metadata

                            new_json[subject_id]["zooniverse subject id"] = subject_id

                        # add in the line of text
                        new_json[subject_id]["text"].append(merged_line)

                        # now add in the aligned individual transcriptions
                        # use the first cluster we found for this line as a "representative cluster"
                        rep_cluster = cluster_set[0]

                        new_json[subject_id]["individual transcriptions"].append(rep_cluster["aligned_text"])

                        # what was the accuracy for this line?
                        accuracy = len([c for c in merged_line if ord(c) != 27])/float(len(merged_line))
                        new_json[subject_id]["accuracy"].append(accuracy)



                        # add in the coordinates
                        # this is only going to work with horizontal lines
                        line_segments = [cluster["center"][:-1] for cluster in cluster_set]
                        x1,x2,y1,y2 = zip(*line_segments)

                        # find the line segments which define the start and end of the line overall
                        x_start = min(x1)
                        x_end = max(x2)

                        start_index = np.argmin(x1)
                        end_index = np.argmax(x2)

                        y_start = y1[start_index]
                        y_end = y1[end_index]

                        new_json[subject_id]["coordinates"].append([x_start,x_end,y_start,y_end])

                # count once per subject
                subjects_with_results += 1
            except KeyError:
                pass

        json.dump(new_json,open("/tmp/"+str(self.project_id)+".json","wb"))

        aws_tar = self.__get_aws_tar_name__()
        with tarfile.open("/tmp/"+aws_tar,mode="w") as t:
            t.add("/tmp/"+str(self.project_id)+".json")

    def __s3_upload__(self):
        s3 = boto3.resource('s3')

        aws_tar = self.__get_aws_tar_name__()

        key = "panoptes-uploads.zooniverse.org/production/project_aggregations_export/"+aws_tar

        s3.Object("zooniverse-static",key).put(Body=open("/tmp/"+aws_tar,"rb"))

    def __get_aws_tar_name__(self):
        media = self.__panoptes_call__("projects/"+str(self.project_id)+"/aggregations_export?admin=true")["media"]
        aggregation_export = media[0]["src"]
        url = aggregation_export.split("?")[0]
        fname = url.split("/")[-1]

        return fname

    def __get_signed_url__(self):
        """
        from http://stackoverflow.com/questions/33549254/how-to-generate-url-from-boto3-in-amazon-web-services
        """
        s3Client = boto3.client('s3')

        aws_tar = self.__get_aws_tar_name__()
        key = "panoptes-uploads.zooniverse.org/production/project_aggregations_export/"+aws_tar

        url = s3Client.generate_presigned_url('get_object', Params = {'Bucket': 'zooniverse-static', 'Key': key}, ExpiresIn = 604800)

        return url

    def __summarize__(self,tar_path=None):
        # start by updating the json output
        self.__restructure_json__()

        # and then upload the files to s3
        self.__s3_upload__()

        # and then get the url to send to people
        url = self.__get_signed_url__()

        # now get some stats to include in the email
        num_retired = self.classification_alg.num_retired
        non_blanks_retired = self.classification_alg.non_blanks_retired
        #
        # stats = self.text_algorithm.stats
        #
        # old_time_string = self.previous_runtime.strftime("%B %d %Y")
        # new_time_string = end_date.strftime("%B %d %Y")
        #
        # accuracy =  1. - stats["errors"]/float(stats["characters"])
        #
        subject = "Aggregation summary for Project " + str(self.project_id) #+ str(old_time_string) + " to " + str(new_time_string)

        body = "This week we have retired " + str(num_retired) + " subjects, of which " + str(non_blanks_retired) + " where not blank."
        # body += " A total of " + str(stats["retired lines"]) + " lines were retired. "
        # body += " The accuracy of these lines was " + "{:2.1f}".format(accuracy*100) + "% - defined as the percentage of characters where at least 3/4's of the users were in agreement."

        body += "\n Greg Hines \n Zooniverse \n \n PS This email was automatically generated."

        # send out the email
        client = boto3.client('ses',region_name='us-east-1')
        response = client.send_email(
            Source='greg@zooniverse.org',
            Destination={
                'ToAddresses': [
                    'greg@zooniverse.org'#,'victoria@zooniverse.org','matt@zooniverse.org'
                ]#,
                # 'CcAddresses': [
                #     'string',
                # ],
                # 'BccAddresses': [
                #     'string',
                # ]
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'ascii'
                },
                'Body': {
                    'Text': {
                        'Data': body,
                        'Charset': 'ascii'
                    }
                }
            },
            ReplyToAddresses=[
                'greg@zooniverse.org',
            ],
            ReturnPath='greg@zooniverse.org'
        )

        print(response)

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"shi:e:d:",["summary","project_id=","environment=","end_date="])
    except getopt.GetoptError:
        warning('transcription.py -i <project_id> -e: <environment> -d: <end_date>')
        sys.exit(2)

    environment = "development"
    project_id = None
    end_date = None
    summary = False

    for opt, arg in opts:
        if opt in ["-i","--project_id"]:
            project_id = int(arg)
        elif opt in ["-e","--environment"]:
            environment = arg
        elif opt in ["-d","--end_date"]:
            end_date = parser.parse(arg)
        elif opt in ["-s","--summary"]:
            summary = True

    assert project_id is not None

    with TranscriptionAPI(project_id,environment,end_date) as project:
        project.__setup__()
        # print "done migrating"
        # # project.__aggregate__(subject_set = [671541,663067,664482,662859])
        processed_subjects = project.__aggregate__()
        # project.__summarize__()

