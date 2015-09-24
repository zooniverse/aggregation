__author__ = 'greg'
import re
import os
import zipfile
import math
import csv
import json
import numpy
import tarfile
import rollbar

class CsvOut:
    def __init__(self,project):
        print type(project)
        # assert isinstance(project,aggregation_api.AggregationAPI)
        self.project = project

        self.project_id = project.project_id
        self.instructions = project.instructions
        self.workflow_names = project.workflow_names
        self.workflows = project.workflows

        self.__yield_aggregations__ = project.__yield_aggregations__
        self.__count_check__ = project.__count_check__
        self.retirement_thresholds = project.retirement_thresholds
        self.versions = project.versions

        # dictionaries to hold the output files
        self.marking_csv_files = {}
        self.classification_csv_files = {}

        self.rollbar_token = project.rollbar_token

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # if another instance is already running - don't do anything, just exit
        # if no error happened - update the timestamp
        # else - the next run will start at the old time stamp (which we want)
        if self.rollbar_token is not None:
            rollbar.init(self.rollbar_token,"production")
            if exc_type is None:
                rollbar.report_message("csv output worked corrected","info")
            else:
                rollbar.report_exc_info()

    def __csv_classification_output__(self,workflow_id,task_id,subject_id,aggregations):
        """
        add a row to the classifciation csv output file
        """
        # first column is the subject id
        row = str(subject_id)

        # now go through each of the possible resposnes
        for answer_index in self.instructions[workflow_id][task_id]["answers"].keys():
            # at some point the integer indices seem to have been converted into strings
            # if a value isn't there - use 0
            if str(answer_index) in aggregations[0].keys():
                row += "," + str(aggregations[0][str(answer_index)])
            else:
                row += ",0"

        # add the number of people who saw this subject
        row += "," + str(aggregations[1])
        self.classification_csv_files[task_id].write(row+"\n")

    def __csv_classification_header_setup__(self,workflow_id,task,output_directory):
        """
        create the csv headers for classification tasks
        :param workflow_id:
        :param task:
        :param output_directory:
        :return:
        """
        fname = self.instructions[workflow_id][task]["instruction"][:50]
        # remove any characters which shouldn't be in a file name
        fname = re.sub(" ","_",fname)
        fname = re.sub("\?","",fname)
        fname = re.sub("\*","",fname)
        fname += ".csv"
        self.classification_csv_files[task] = open(output_directory+fname,"wb")
        header = "subject_id"
        for answer_index in sorted(self.instructions[workflow_id][task]["answers"].keys()):
            answer = self.instructions[workflow_id][task]["answers"][answer_index]
            answer = re.sub(",","",answer)
            answer = re.sub(" ","_",answer)
            header += ",p("+answer+")"
        header += ",num_users"
        self.classification_csv_files[task].write(header+"\n")

    def __csv_file_setup__(self,workflow_id):
        """
        open csv files for each output and write headers for each file
        """
        # close any previously opened files - needed when we have multiple workflows per project
        for f in self.marking_csv_files.values():
            assert isinstance(f,file)
            f.close()

        for f in self.classification_csv_files.values():
            assert isinstance(f,file)
            f.close()

        # and reset
        self.marking_csv_files = {}
        self.classification_csv_files = {}

        # start by creating a directory specific to this project
        output_directory = "/tmp/"+str(self.project_id)+"/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # now make the directory specific to the workflow
        # first - remove any bad characters
        workflow_name = self.workflow_names[workflow_id]
        workflow_name = re.sub(" ","_",workflow_name)

        output_directory += workflow_name +"/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # create headers to eachcsv file
        classification_tasks,marking_tasks = self.workflows[workflow_id]
        for task in marking_tasks:
            self.__csv_marking_header_setup__(workflow_id,task,set(marking_tasks[task]),output_directory)

        for task in classification_tasks:
            self.__csv_classification_header_setup__(workflow_id,task,output_directory)
        #     # use the instruction label to create the csv file name
        #     # todo - what if the instruction labels are the same?
        #     fname = self.instructions[workflow_id][task]["instruction"][:50]
        #
        #     # remove any characters which shouldn't be in a file name
        #     fname = re.sub(" ","_",fname)
        #     fname = re.sub("\?","",fname)
        #     fname = re.sub("\*","",fname)
        #     fname += ".csv"
        #     self.classification_csv_files[task] = open(output_directory+fname,"wb")
        #     header = "subject_id"
        #     for answer_index in sorted(self.instructions[workflow_id][task]["answers"].keys()):
        #         answer = self.instructions[workflow_id][task]["answers"][answer_index]
        #         answer = re.sub(",","",answer)
        #         answer = re.sub(" ","_",answer)
        #         header += ",p("+answer+")"
        #     header += ",num_users"
        #     self.classification_csv_files[task].write(header+"\n")

    def __write_out__(self,subject_set = None,compress=True):
        """
        create the csv outputs for a given set of workflows
        the workflows are specified by self.workflows which is determined when the aggregation engine starts
        a zipped file is created in the end
        """
        assert (subject_set is None) or isinstance(subject_set,int)

        tarball = None
        if compress:
            tarball = tarfile.open("/tmp/"+str(self.project_id)+"export.tar.gz", "w:gz")

        for workflow_id in self.workflows:
            print "csv output for workflow - " + str(workflow_id)
            self.__csv_file_setup__(workflow_id)
            classification_tasks,marking_tasks = self.workflows[workflow_id]
            print classification_tasks
            print marking_tasks

            for subject_id,task_id,aggregations in self.__yield_aggregations__(workflow_id,subject_set):
                # check to see if the correct number of classifications were received
                # todo - this is only a stop gap measure until we figure out why some subjects are being
                # todo - retired early. Once that is done, we can remove this
                if self.__count_check__(workflow_id,subject_id) < self.retirement_thresholds[workflow_id]:
                    print "skipping"
                    continue

                # are there markings associated with this task?
                if task_id in marking_tasks:
                    for shape in set(marking_tasks[task_id]):
                        if shape == "polygon":
                            self.__polygon_summary_output__(workflow_id,task_id,subject_id,aggregations)
                            self.__polygon_heatmap_output__(workflow_id,task_id,subject_id,aggregations)
                    # self.__csv_marking__output__(workflow_id,task_id,subject_id,aggregations,marking_tasks[task_id])

                # are there any classifications associated with this task
                if task_id in classification_tasks:
                    self.__csv_classification_output__(workflow_id,task_id,subject_id,aggregations)

        # finally zip everything (over all workflows) into one zip file
        # self.__csv_to_zip__()
        if compress:
            tarball.close()
            return "/tmp/"+str(self.project_id)+"export.tar.gz"

    # def __csv_annotations__(self,workflow_id_filter,subject_set):
    #     # find the major id of the workflow we are filtering
    #     version_filter = int(math.floor(float(self.versions[workflow_id_filter])))
    #
    #     if subject_set is None:
    #         subject_set = self.__load_subjects__(workflow_id_filter)
    #
    #     with open(self.csv_classification_file, 'rb') as csvfile:
    #         reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #
    #         for row in reader:
    #             subject_data = row[8]
    #             annotations = row[7]
    #             workflow_id = row[2]
    #             workflow_version = row[4]
    #
    #             # convert to json form
    #             subject_data = json.loads(subject_data)
    #             subject_id = subject_data.keys()[0]
    #
    #             # csv file contains classifications from every workflow - so make sure we find
    #             # only the one we currently want
    #             if int(workflow_id) != workflow_id_filter:
    #                 continue
    #
    #             # if these are not one of the subjects we are looking for
    #             if subject_id not in subject_set:
    #                 continue
    #
    #             # convert to float
    #             workflow_version = float(workflow_version)
    #             # if we are not at the correct major version id, skip
    #             if workflow_version < version_filter:
    #                 continue

    def __csv_marking_header_setup__(self,workflow_id,task,tools,output_directory):
        """
        tools - says what sorts of different types of shapes/tools we have to do deal with for this task
        we can either give the output for each tool in a completely different csv file - more files, might
        be slightly overwhelming, but then we could make the column headers more understandable
        """
        if "polygon" in tools:
            key = task+"polygon_summary"
            self.marking_csv_files[key] = open(output_directory+task+"_polygons_summary.csv","wb")
            header = "subject_id,num_users,minimum_users_per_cluster,area(noise),tool_certainity"
            for tool_id in sorted(self.instructions[workflow_id][task]["tools"].keys()):
                tool = self.instructions[workflow_id][task]["tools"][tool_id]["marking tool"]
                tool = re.sub(" ","_",tool)
                header += ",area("+tool+")"
            self.marking_csv_files[key].write(header+"\n")

            key = task+"polygon_heatmap"
            self.marking_csv_files[key] = open(output_directory+task+"_polygons_heatmap.csv","wb")
            header = "subject_id,num_users,pts"
            self.marking_csv_files[key].write(header+"\n")


        # print workflow_id
        # print task
        # assert False
        # # build up the header row
        # header = "subject_id"
        # for tool_id in sorted(self.instructions[workflow_id][task]["tools"].keys()):
        #     tool = self.instructions[workflow_id][task]["tools"][tool_id]["marking tool"]
        #     header += ","+tool
        # header += ",mean probability,median probability,mean tool likelihood,median tool likelihood,number of users"
        # self.marking_csv_files[task].write(header+"\n")

    def __polygon_heatmap_output__(self,workflow_id,task_id,subject_id,aggregations):
        """
        print out regions according to how many users selected that user - so we can a heatmap
        of the results
        :param workflow_id:
        :param task_id:
        :param subject_id:
        :param aggregations:
        :return:
        """
        key = task_id+"polygon_heatmap"
        for cluster_index,cluster in aggregations["polygon clusters"].items():
            # each cluster refers to a specific tool type - so there can actually be multiple blobs
            # (or clusters) per cluster
            # not actually clusters

            if cluster_index in ["param","all_users"]:
                continue

            if cluster["tool classification"] is not None:
                # this result is not relevant to the heatmap
                continue

            row = str(subject_id) + "," + str(cluster["num users"]) + ",\"" + str(cluster["center"]) + "\""
            self.marking_csv_files[key].write(row+"\n")

    def __polygon_summary_output__(self,workflow_id,task_id,subject_id,aggregations):
        """
        print out a csv summary of the polygon aggregations (so not the individual xy points)
        need to know the workflow and task id so we can look up the instructions
        that way we can know if there is no output for a given tool - that tool wouldn't appear
        at all in the aggregations
        """
        # find out which tools actually corresponds to polygons - they could correspond to other tools/shapes
        marking_shapes = self.workflows[workflow_id][1][task_id]
        polygon_tools = [tool_id for tool_id,shape in enumerate(marking_shapes) if shape == "polygon"]

        area_per_type = {}#t:0 for t in polygon_tools}
        certainty_per_type = {}#t: -1 for t in polygon_tools}

        row = str(subject_id)
        # if noise_area stays 0, that means that there wasn't any noise at all :)
        noise_area = 0
        num_users = 0
        for cluster_index,cluster in aggregations["polygon clusters"].items():
            # each cluster refers to a specific tool type - so there can actually be multiple blobs
            # (or clusters) per cluster
            # not actually clusters
            if cluster_index == "all_users":
                num_users = len(cluster)
                continue

            if cluster_index in ["param","all_users"]:
                continue

            if cluster["tool classification"] is None:
                # this result is not relevant to the summary stats
                continue

            # this value will just get repeatedly read in - which is fine
            noise_area = cluster["incorrect area"]

            # cluster = -1 => empty image
            if cluster["certainty"] >= 0:
                most_likely_type = cluster["tool classification"]
                area_per_type[most_likely_type] = cluster["area"]
                certainty_per_type[most_likely_type] = cluster["certainty"]

        row += ","+str(num_users)
        # todo - don't hard code this
        row += ",3"
        row += "," + str(noise_area)

        # calculate the overall (weighted) certainty
        area = [area_per_type[t] for t in polygon_tools if t in area_per_type]
        certainty = [certainty_per_type[t] for t in polygon_tools if t in certainty_per_type]
        assert len(area) == len(certainty)
        if area != []:
            weighted_overall_certainty = numpy.average(certainty,weights =area)
        else:
            weighted_overall_certainty = "NA"

        row += ","+str(weighted_overall_certainty)

        for t in polygon_tools:
            if t in area_per_type:
                row += ","+str(area_per_type[t])
            else:
                row += ",0"

        key = task_id+"polygon_summary"
        self.marking_csv_files[key].write(row+"\n")

    def __csv_to_zip__(self):
        """
        put the results into a  nice csv file
        """
        # code taken from http://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
        zipf = zipfile.ZipFile("/tmp/"+str(self.project_id)+".zip", 'w')

        # walk through the output directory, compressing as we go
        for root, dirs, files in os.walk("/tmp/"+str(self.project_id)+"/"):
            for file in files:
                zipf.write(os.path.join(root, file))

        zipf.close()