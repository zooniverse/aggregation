from __future__ import print_function
from aggregation_api import AggregationAPI
import pandas
import classification
import yaml
import parser
import json
import math
import csv_output

def extract_subject_id(subject_data):
    """
    extract the subject id for each subject
    :param subject_data:
    :return:
    """
    json_subject = json.loads(subject_data)
    return int(json_subject.keys()[0])

def load_json(json_string):
    return json.loads(json_string)

class LocalAggregationAPI(AggregationAPI):
    def __init__(self,project_id,csv_classification_file):
        AggregationAPI.__init__(self,project_id,"development")

        # read in the csv file as a dataframe (pandas)
        self.classifications_dataframe = pandas.read_csv(csv_classification_file)
        # extract the subject id for each subject - based on the subject data field
        self.classifications_dataframe["subject_id"] = self.classifications_dataframe["subject_data"].map(lambda x: extract_subject_id(x))

        self.aggregation_results = {}

    def __setup__(self):
        """
        set up all the connections to panoptes and different databases
        :return:
        """
        print("setting up")
        # just for when we are treating an ouroboros project like a panoptes one
        # in which the subject ids will be zooniverse_ids, which are strings
        self.subject_id_type = "int"

        # todo - some time in the far future - complete support for expert annotations
        self.experts = []

        # load in the project id (a number)
        # if one isn't provided, tried reading it from the yaml config file
        # todo - if we are not using a secure connection, this forces the project_id
        # todo - to be provided as a param (since we don't read in the yaml file)
        # todo - probably want to change that at some point as there is value in
        # todo - non logged in users having a yaml file (where they can give csv classification files etc.)

        #########
        # everything that follows assumes you have a secure connection to Panoptes
        # plus the DBs (either production or staging)

        param_file = open("/app/config/aggregation.yml","rb")
        param_details = yaml.load(param_file)

        environment_details = param_details[self.environment]
        # do we have a specific date as the minimum date for this project?
        if (self.project_id in param_details) and ("default_date" in param_details[self.project_id]):
            self.previous_runtime = parser.parse(param_details[self.project_id]["default_date"])

        print("trying secure Panoptes connection")
        self.__panoptes_connect__(environment_details)

        self.__get_project_details__()

        # todo - refactor all this?
        # there may be more than one workflow associated with a project - read them all in
        # and set up the associated tasks
        self.workflows,self.versions,self.instructions,self.updated_at_timestamps = self.__get_workflow_details__()
        print("workflows are " + str(self.workflows))
        self.retirement_thresholds = self.__get_retirement_threshold__()
        self.workflow_names = self.__get_workflow_names__()

        # set up the clustering algorithms
        self.__setup_clustering_algs__()
        # load the default classification algorithm
        self.__set_classification_alg__(classification.VoteCount)

        # a bit of a sanity check in case I forget to change back up before uploading
        # production and staging should ALWAYS pay attention to the version and only
        # aggregate retired subjects
        if self.environment in ["production","staging"]:
            self.only_retired_subjects = True

        # bit of a stop gap measure - stores how many people have classified a given subject
        self.classifications_per_subject = {}

        # do we want to aggregate over only retired subjects?

        # do we want to aggregate over only subjects that have been retired/classified since
        # the last time we ran the code?
        self.only_recent_subjects = False

    def __migrate__(self,workflow_id,version,subject_set=None):
        """
        since we don't actually have to migrate classifications between databases, just return the set of all
        subjects which have had classifications
        :param workflow_id:
        :param version:
        :param subject_set:
        :return:
        """
        data_frame = self.classifications_dataframe

        return set(data_frame["subject_id"])

    def __yield_annotations__(self,workflow_id,subject_set):
        """
        get all of the annotations for this particular workflow id and each subject in this subject set
        :param workflow_id:
        :param subject_set:
        :return:
        """
        data_frame = self.classifications_dataframe

        for subject_id in subject_set:
            # select only those annotations for the current subject id
            data_frame = data_frame[(data_frame.subject_id==int(subject_id)) & (data_frame.workflow_id == workflow_id)]

            # what is the current workflow version for this particular workflow id
            version = int(math.floor(float(self.versions[workflow_id])))
            data_frame = data_frame[(data_frame.workflow_version >= version)]

            users_per_subjects = data_frame["user_id"]
            annotations_per_subjects = data_frame["annotations"]

            # todo - load image dimensions
            yield int(subject_id),users_per_subjects,annotations_per_subjects,(None,None)

        raise StopIteration()

    def __get_previously_aggregated__(self,workflow_id):
        """
        only useful for doing upserts - ie in production mode
        :param workflow_id:
        :return:
        """
        return None

    def __upsert_results__(self,workflow_id,aggregations,previously_aggregated):
        """
        store the results in a dictionary - the term upsert only makes sense if we are storing to a db
        :param workflow_id:
        :param aggregations:
        :param previously_aggregated:
        :return:
        """
        # convert to int to be safe
        workflow_id = int(workflow_id)
        print("upserting " + str(workflow_id))
        if workflow_id not in self.aggregation_results:
            self.aggregation_results[workflow_id] = dict()
        for subject_id,agg in aggregations.items():
            self.aggregation_results[workflow_id][subject_id] = agg

    def __count_subjects_classified__(self,workflow_id):
        """
        return a count of all the subjects classified
        :param workflow_id:
        :return:
        """
        workflow_id = int(workflow_id)

        # if we haven't saved any aggregations - the total is 0
        if workflow_id not in self.aggregation_results:
            return 0
        else:
            return len(self.aggregation_results[int(workflow_id)])

    def __yield_aggregations__(self,workflow_id,subject_set=None):
        """
        return all of the aggregations for the given workflow_id/subject_set
        :param workflow_id:
        :param subject_set:
        :return:
        """
        workflow_id = int(workflow_id)

        # stop if we don't have any aggregations
        if workflow_id not in self.aggregation_results:
            raise StopIteration()

        for subject_id,aggregation in self.aggregation_results[workflow_id].items():
            # if we have provided a filter for the subject ids and the current id is not in the filter
            # skip this aggregation
            if (subject_set is not None) and (subject_id not in subject_set):
                continue

            yield subject_id,aggregation

        raise StopIteration()

if __name__ == "__main__":
    project = LocalAggregationAPI(63,"/home/ggdhines/Downloads/copy-of-kitteh-zoo-subjects-classifications.csv")
    project.__setup__()
    project.__aggregate__()

    with csv_output.CsvOut(project) as c:
        c.__write_out__()