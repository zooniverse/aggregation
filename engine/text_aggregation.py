#!/usr/bin/env python
from __future__ import print_function
from aggregation_api import AggregationAPI
from classification import Classification
import rollbar
import requests
import datetime
import helper_functions
import yaml
import json
from rectangle_clustering import RectangleClustering
import parser
import getopt
import sys
import boto3
from helper_functions import warning
import os
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from transcription_output import ShakespearesWorldOutput,AnnotateOutput
__author__ = 'ggdhines'


class SubjectRetirement(Classification):
    def __init__(self,environment,project):
        Classification.__init__(self,environment)

        self.project = project

        self.num_retired = None
        self.non_blanks_retired = None
        # to know how often we should call Panoptes to get a new token
        # save on having to make unnecessary calls
        self.token_date = datetime.datetime.now()
        self.to_retire = set()

        self.total_retired = 0

    def __get_blank_subjects__(self,raw_classifications):
        """
        get all subjects which people have identified as blank
        :param raw_classifications:
        :return:
        """
        blank_subjects = set()

        for subject_id in raw_classifications["T0"]:
            user_ids,is_subject_empty = zip(*raw_classifications["T0"][subject_id])
            if is_subject_empty != []:
                empty_count = sum([1 for i in is_subject_empty if i == True])
                if empty_count >= 3:
                    blank_subjects.add(subject_id)

        return blank_subjects

    def __get_completed_subjects__(self,raw_classifications):
        """
        return all the subjects which people have said are completely transcribed
        :param raw_classifications:
        :return:
        """
        completed_subjects = set()

        for subject_id in raw_classifications["T3"]:
            user_ids,completely_transcribed = zip(*raw_classifications["T3"][subject_id])

            completely_count = sum([1 for i in completely_transcribed if i == True])
            if completely_count >= 3:
                completed_subjects.add(subject_id)

        return completed_subjects

    def __aggregate__(self,raw_classifications,workflow,aggregations,workflow_id):
        """
        classification aggregation for annotate/folger means looking for subjects which we can retire
        :param raw_classifications:
        :param workflow:
        :param aggregations:
        :param workflow_id:
        :return:
        """
        assert isinstance(workflow_id,int)
        to_retire = set()
        # start by looking for empty subjects
        # "T0" really should always be there but we may have a set of classifications (really old ones before
        # the workflow changed) where it is missing - if "T0" isn't there, just skip
        if "T0" in raw_classifications:
            to_retire.update(self.__get_blank_subjects__(raw_classifications))

        # now look to see what has been completely transcribed
        if "T3" in raw_classifications:
            to_retire.update(self.__get_completed_subjects__(raw_classifications))

        # call the Panoptes API to retire these subjects
        # get an updated token
        time_delta = datetime.datetime.now()-self.token_date
        # update every 30 minutes
        if time_delta.seconds > (30*60):
            self.token_date = datetime.datetime.now()
            assert isinstance(self.project,AggregationAPI)
            self.project.__panoptes_connect__()

        token = self.project.token

        # print("retiring " + str(len(to_retire)) + " subjects")

        # need to retire the subjects one by one
        for retired_subject in to_retire:
            self.total_retired += 1
            self.to_retire.add(retired_subject)
            try:
                headers = {"Accept":"application/vnd.api+json; version=1","Content-Type": "application/json", "Authorization":"Bearer "+token}
                params = {"subject_id":retired_subject}
                r = requests.post("https://panoptes.zooniverse.org/api/workflows/"+str(workflow_id)+"/retired_subjects",headers=headers,data=json.dumps(params))
                # if self.environment != "production":
                #     print(r.status_code)
                # return an error if we have a 404 status
                assert r.status_code == 200
            except TypeError as e:
                warning(e)
                rollbar.report_exc_info()

        # if to_retire != set():
        #     print("total retired so far " + str(self.total_retired))

        # print("we would have retired " + str(len(self.to_retire)))
        # print("with non-blanks " + str(len(self.to_retire)-blank_retirement))
        # print(str(len(self.to_retire)-blank_retirement))
        #
        # self.num_retired = len(self.to_retire)
        # self.non_blanks_retired = len(self.to_retire)-blank_retirement

        return aggregations


class TranscriptionAPI(AggregationAPI):
    def __init__(self,project_id,environment,end_date=None):
        AggregationAPI.__init__(self,project_id,environment,end_date=end_date)

        # just to stop me from using transcription on other projects
        assert int(project_id) in [245,376]

    def __cluster__(self,used_shapes,raw_markings,image_dimensions,aggregations):
        """
        :param aggregations: we're working on a subject by subject basis - aggregations is from previous subjects
        """

        if raw_markings == {}:
            warning("skipping")
            return aggregations

        # start by clustering text
        # print("clustering text")
        cluster_aggregations = self.text_algorithm.__aggregate__(raw_markings,image_dimensions)
        aggregations = self.__merge_aggregations__(aggregations,cluster_aggregations)
        # print("clustering images")
        image_aggregations = self.image_algorithm.__aggregate__(raw_markings,image_dimensions)

        aggregations = self.__merge_aggregations__(aggregations,image_aggregations)

        return aggregations

    def __setup__(self):
        """
        do setup specifically for annotate and shakespeare's world.
        things like using a special classification algorithm (which is able to retire subject) and
        text clustering algorithms specifically designed for annotate/shakespeare's world
        :return:
        """
        AggregationAPI.__setup__(self)

        workflow_id = self.workflows.keys()[0]

        # set the classification algorithm which will retire the subjects
        self.__set_classification_alg__(SubjectRetirement,self)

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

        self.email_recipients = api_details[self.project_id]["email"]

        # now that we have the additional text arguments, convert text_algorithm from a class
        # to an actual instance
        if self.project_id == 245:
            import annotate
            self.text_algorithm = annotate.AnnotateClustering("text",self,additional_text_args)
            self.output_tool = AnnotateOutput(self)
        elif self.project_id == 376:
            import folger
            self.text_algorithm = folger.FolgerClustering("text",self,additional_text_args)
            self.output_tool = ShakespearesWorldOutput(self)
        else:
            assert False

        self.image_algorithm = RectangleClustering("image",self,{})


        self.only_retired_subjects = False
        self.only_recent_subjects = True

    def __exit__(self, exc_type, exc_value, traceback):
        """
        report any errors via rollbar and shut down
        :param exc_type:
        :param exc_value:
        :param traceback:
        :return:
        """
        if (exc_type is not None) and (self.environment == "production"):
            panoptes_file = open("/app/config/aggregation.yml","rb")
            api_details = yaml.load(panoptes_file)

            rollbar_token = api_details[self.environment]["rollbar"]
            rollbar.init(rollbar_token,self.environment)
            rollbar.report_exc_info()

        # calling the parent
        AggregationAPI.__exit__(self, exc_type, exc_value, traceback)


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



    def __s3_connect__(self):
        """
        connect to s3 - currently return both S3Connection and client because they seem
        to do offer different functionality - uploading files vs. generating signed urls
        seems pretty silly that this is the case - so feel free to fix it
        :return:
        """
        # Adam has created keys which always work - had trouble with sending out emails otherwise
        param_file = open("/app/config/aws.yml","rb")
        param_details = yaml.load(param_file)

        id_ = param_details["aws_access_key_id"]
        key = param_details["aws_secret_access_key"]

        conn = S3Connection(id_,key)

        # s3 = boto3.resource("s3",aws_access_key_id=id_,aws_secret_access_key=key)

        client = boto3.client(
            's3',
            aws_access_key_id=id_,
            aws_secret_access_key=key,
        )

        return conn,client

    def __s3_upload__(self):
        """
        upload the file to s3
        see http://boto.cloudhackers.com/en/latest/s3_tut.html
        :return:
        """
        # s3 = boto3.resource('s3')
        s3,_ = self.__s3_connect__()

        aws_tar = self.__get_aws_tar_name__()

        b = s3.get_bucket('zooniverse-static')

        key_str = "panoptes-uploads.zooniverse.org/production/project_aggregations_export/"+aws_tar

        s3_key = Key(b)
        s3_key.key = key_str

        if not os.path.exists("/tmp/"+aws_tar):
            print("warning the tar file does not exist - creating an temporary one.")
            panoptes_file = open("/app/config/aggregation.yml","rb")
            api_details = yaml.load(panoptes_file)

            rollbar_token = api_details[self.environment]["rollbar"]
            rollbar.init(rollbar_token,self.environment)
            rollbar.report_message('the tar file does not exist', 'warning')
            with open("/tmp/"+aws_tar,"w") as f:
                f.write("")

        s3_key.set_contents_from_filename("/tmp/"+aws_tar)

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
        # s3Client = boto3.client('s3')
        _,s3 = self.__s3_connect__()

        aws_tar = self.__get_aws_tar_name__()
        key = "panoptes-uploads.zooniverse.org/production/project_aggregations_export/"+aws_tar

        url = s3.generate_presigned_url('get_object', Params = {'Bucket': 'zooniverse-static', 'Key': key}, ExpiresIn = 604800)

        return url

    def __summarize__(self,tar_path=None):
        # start by updating the json output
        self.output_tool.__json_output__()

        # and then upload the files to s3
        self.__s3_upload__()

        # and then get the url to send to people
        url = self.__get_signed_url__()



        subject = "Aggregation summary for Project " + str(self.project_id) #+ str(old_time_string) + " to " + str(new_time_string)

        body = "The link to the json aggregation results for all retired subjects is " + url
        # body += " A total of " + str(stats["retired lines"]) + " lines were retired. "
        # body += " The accuracy of these lines was " + "{:2.1f}".format(accuracy*100) + "% - defined as the percentage of characters where at least 3/4's of the users were in agreement."

        body += "\n\n Greg Hines \n Zooniverse \n \n PS This email was automatically generated."

        # send out the email
        client = boto3.client('ses',region_name='us-east-1')
        response = client.send_email(
            Source='greg@zooniverse.org',
            Destination={
                'ToAddresses': [
                    'greg@zooniverse.org',self.email_recipients
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
        print("response from emailing results")
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
        # project.__reset_cassandra_dbs__()
        # print "done migrating"
        # # project.__aggregate__(subject_set = [671541,663067,664482,662859])

        # for workflow_id,version in project.versions.items():
        #     print(workflow_id)
        #     migrated_subjects = project.__migrate__(workflow_id,version)
        # project.output_tool.__json_output__()

        # processed_subjects = project.__aggregate__()
        # print("about to send off email")
        if datetime.datetime.today().weekday() == 1:#environment == "production":
            project.__summarize__()

