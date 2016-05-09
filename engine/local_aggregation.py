from __future__ import print_function
from aggregation_api import AggregationAPI
import pandas
import classification
import yaml
import parser

class LocalAggregationAPI(AggregationAPI):
    def __init__(self,csv_classification_file):
        AggregationAPI.__init__(self,63,"development")

        # todo - allow users to provide their own classification and subject files
        self.csv_classification_file = csv_classification_file

        self.classifications_dataframe = pandas.read_csv(csv_classification_file)

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
        self.retirement_thresholds = self.__get_retirement_threshold__()
        self.workflow_names = self.__get_workflow_names__()

        # is there an entry for the project in the yaml file?
        # if so, has a specific workflow id has been provided?
        # todo - this can be removed or rewritten
        if "workflow_id" in environment_details:
            workflow_id = int(environment_details["workflow_id"])
            try:
                print("aggregating only for workflow id : " + str(workflow_id))
                self.workflows = {workflow_id: self.workflows[workflow_id]}
            except KeyError:
                warning("did not have given desired workflow: " + str(workflow_id))
                warning("here's the workflows we do have")
                warning(self.workflows)
                raise

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
        data_frame = self.classifications_dataframe
        print(data_frame[data_frame["workflow_id"] == workflow_id])
        print(workflow_id)
        assert False

    def __yield_annotations__(self,workflow_id,subject_set):
        print(subject_set)
        assert False

if __name__ == "__main__":
    engine = LocalAggregationAPI("/home/ggdhines/Downloads/copy-of-kitteh-zoo-subjects-classifications.csv")
    engine.__setup__()
    engine.__aggregate__()
