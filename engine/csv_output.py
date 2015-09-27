__author__ = 'greg'
import re
import os
import numpy
import tarfile
import shutil

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

        # dictionary to hold the output files
        self.csv_files = {}

        self.rollbar_token = project.rollbar_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __classification_output__(self,workflow_id,task_id,subject_id,aggregations):
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
        self.csv_files[task_id].write(row+"\n")

    def __classification_header_setup__(self,workflow_id,task,output_directory):
        """
        create the csv headers for classification tasks
        :param workflow_id:
        :param task:
        :param output_directory:
        :return:
        """
        fname = str(task) + self.instructions[workflow_id][task]["instruction"][:50]
        fname = self.__csv_string__(fname)
        fname += ".csv"
        self.csv_files[task] = open(output_directory+fname,"wb")
        header = "subject_id"
        for answer_index in sorted(self.instructions[workflow_id][task]["answers"].keys()):
            answer = self.instructions[workflow_id][task]["answers"][answer_index]
            answer = self.__csv_string__(answer)
            header += ",p("+answer+")"
        header += ",num_users"

        self.csv_files[task].write(header+"\n")

    def __followup_header_setup__(self,workflow_id,task,tool,followup_index,output_directory):
        followup_question = self.instructions[workflow_id][task]["tools"][tool]["followup_questions"][followup_index]

        fname = str(task) + "_" + str(tool) + "_"+str(followup_index)+followup_question["question"][:25]
        fname = self.__csv_string__(fname)
        fname += ".csv"
        self.csv_files[(task,tool,followup_index)] = open(output_directory+fname,"wb")

        header = "subject_id,cluster_index"
        for answer_index in sorted(followup_question["answers"].keys()):
            answer = followup_question["answers"][answer_index]["label"]
            answer = self.__csv_string__(answer)
            header += ",p("+answer+")"

        header += ",num_users"
        self.csv_files[(task,tool,followup_index)].write(header+"\n")

    def __files_setup__(self,workflow_id):
        """
        open csv files for each output and write headers for each file
        """
        # and reset
        self.csv_files = {}

        # now make the directory specific to the workflow
        # first - remove any bad characters
        workflow_name = self.workflow_names[workflow_id]
        workflow_name = self.__csv_string__(workflow_name)

        output_directory = "/tmp/"+str(self.project_id)+"/"
        output_directory += workflow_name +"/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # create headers to eachcsv file
        classification_tasks,marking_tasks = self.workflows[workflow_id]
        for task in marking_tasks:
            self.__marking_header_setup__(workflow_id,task,set(marking_tasks[task]),output_directory)

        for task in classification_tasks:
            if isinstance(classification_tasks[task],bool):
                print "creating header for classification task " + str(task)
                self.__classification_header_setup__(workflow_id,task,output_directory)
            else:
                print "creating headers for followup task " + str(task)

                for tool in classification_tasks[task]:
                    for followup_index in classification_tasks[task][tool]:
                        self.__followup_header_setup__(workflow_id,task,tool,followup_index,output_directory)

    def __followup_output__(self,workflow_id,task_id,subject_id,aggregations):
        classification_tasks,marking_tasks = self.workflows[workflow_id]

        for tool in classification_tasks[task_id]:
            # what shape does this tool make?
            shape = marking_tasks[task_id][tool]

            # now go through each of the clusters - and find the relevant ones
            for cluster_index in sorted(aggregations[shape + " clusters"].keys()):
                if cluster_index == "all_users":
                    continue
                cluster = aggregations[shape + " clusters"][cluster_index]
                # what tool was this cluster most likely made with? (or should have been made with)
                most_likely_tool,_ = max(cluster["tool_classification"][0].items(),key = lambda x:x[1])

                # if the tool is the one associated with the follow up questions we are currently interested in
                if int(most_likely_tool) == int(tool):
                    # go through each follow up question
                    for question_index in classification_tasks[task_id][tool]:
                        # rely on the original instructions since some of th values might not appear in the results
                        answer_range = sorted(self.instructions[workflow_id][task_id]["tools"][tool]["followup_questions"][question_index]["answers"].keys())

                        row = str(subject_id)+","+str(cluster_index)

                        # now go through each of the possible resposnes
                        for answer_index in answer_range:
                            # at some point the integer indices seem to have been converted into strings
                            # if a value isn't there - use 0

                            if str(answer_index) in cluster["followup_question"][str(question_index)][0]:
                                row += "," + str(cluster["followup_question"][str(question_index)][0][str(answer_index)])
                            else:
                                row += ",0"

                        # add the number of people who saw this subject
                        row += "," + str(cluster["followup_question"][str(question_index)][1])

                        self.csv_files[(task_id,tool,question_index)].write(row+"\n")

                # print aggregations[subject_id]




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

        # start by creating a directory specific to this project
        output_directory = "/tmp/"+str(self.project_id)+"/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # move over the readme and add it to the tar file
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        shutil.copy(curr_dir+"/readme.txt","/tmp/"+str(self.project_id)+"/")
        with open("/tmp/"+str(self.project_id)+"/readme.txt", "rb") as readfile:
            tarInfo = tarball.gettarinfo(fileobj=readfile)
            tarball.addfile(tarInfo, fileobj=readfile)

        for workflow_id in self.workflows:
            print "csv output for workflow - " + str(workflow_id)
            self.__files_setup__(workflow_id)
            classification_tasks,marking_tasks = self.workflows[workflow_id]

            for subject_id,task_id,aggregations in self.__yield_aggregations__(workflow_id,subject_set):
                # check to see if the correct number of classifications were received
                # todo - this is only a stop gap measure until we figure out why some subjects are being
                # todo - retired early. Once that is done, we can remove this
                # if self.__count_check__(workflow_id,subject_id) < self.retirement_thresholds[workflow_id]:
                #     # print "skipping"
                #     continue

                # are there markings associated with this task?
                if task_id in marking_tasks:
                    for shape in set(marking_tasks[task_id]):
                        if shape == "polygon":
                            self.__polygon_summary_output__(workflow_id,task_id,subject_id,aggregations)
                            self.__polygon_heatmap_output__(workflow_id,task_id,subject_id,aggregations)
                        # the following shapes can basically be dealt with in the same way
                        elif shape in ["point","line"]:
                            self.__marking_output__(workflow_id,task_id,subject_id,aggregations,shape)
                            # if we are only using the point marking for people to count items (and you don't
                            # care about the xy coordinates) - the function below will give you what you want
                            self.__shape_summary_output__(workflow_id,task_id,subject_id,aggregations,shape)
                        elif shape == "rectangle":
                            # todo - finish this part
                            self.__rectangle_output__(workflow_id,task_id,subject_id,aggregations)
                        else:
                            print shape
                            assert False

                # are there any classifications associated with this task
                if task_id in classification_tasks:
                    # normal output
                    if isinstance(classification_tasks[task_id],bool):
                        self.__classification_output__(workflow_id,task_id,subject_id,aggregations)
                    else:
                        self.__followup_output__(workflow_id,task_id,subject_id,aggregations)

            for fname,f in self.csv_files.items():
                assert isinstance(f,file)
                if compress:
                    print "writing out " + str(fname)
                    f.close()
                    with open(f.name, "rb") as readfile:
                        tarInfo = tarball.gettarinfo(fileobj=readfile)
                        tarball.addfile(tarInfo, fileobj=readfile)
                f.close()



        # finally zip everything (over all workflows) into one zip file
        # self.__csv_to_zip__()
        if compress:
            tarball.close()
            return "/tmp/"+str(self.project_id)+"export.tar.gz"

    # todo - figure out if this string is necessary
    def __csv_string__(self,string):
        """
        remove or replace all characters which might cause problems in a csv template
        :param str:
        :return:
        """
        assert isinstance(string,str) or isinstance(string,unicode)
        string = re.sub(" ","_",string)
        string = re.sub("\.","",string)
        string = re.sub("#","",string)
        string = re.sub("\(","",string)
        string = re.sub("\)","",string)
        string = re.sub("\?","",string)
        string = re.sub("\*","",string)
        string = re.sub("-","",string)
        string = re.sub("/","",string)
        string = re.sub(":","",string)
        string = re.sub("\"","",string)
        string = re.sub("%","",string)

        return string

    def __marking_output__(self,workflow_id,task_id,subject_id,aggregations,shape):
        """
        output for line segments
        :param workflow_id:
        :param task_id:
        :param subject_id:
        :param aggregations:
        :return:
        """
        key = task_id + shape
        for cluster_index,cluster in aggregations[shape + " clusters"].items():
            if cluster_index == "all_users":
                continue

            # build up the row bit by bit to have the following structure
            # "subject_id,most_likely_tool,x,y,p(most_likely_tool),p(true_positive),num_users"
            row = str(subject_id)+","

            # extract the most likely tool for this particular marking and convert it to
            # a string label
            tool_classification = cluster["tool_classification"][0].items()
            most_likely_tool,tool_probability = max(tool_classification, key = lambda x:x[1])
            tool_str = self.instructions[workflow_id][task_id]["tools"][int(most_likely_tool)]["marking tool"]
            row += tool_str + ","

            # get the central coordinates next
            for center_param in cluster["center"]:
                row += str(center_param) + ","

            # add on how likely the most likely tool was
            row += str(tool_probability) + ","
            # how likely the cluster is to being a true positive and how many users (out of those who saw this
            # subject) actually marked it. For the most part p(true positive) is equal to the percentage
            # of people, so slightly redundant but allows for things like weighted voting and IBCC in the future
            prob_true_positive = cluster["existence"][0]["1"]
            num_users = cluster["existence"][1]
            row += str(prob_true_positive) + "," + str(num_users)
            self.csv_files[key].write(row+"\n")

    def __rectangle_output__(self,workflow_id,task_id,subject_id,aggregations):
        key = task_id + "rectangle"
        for cluster_index,cluster in aggregations["rectangle clusters"].items():
            if cluster_index == "all_users":
                continue

            # build up the row bit by bit to have the following structure
            # "subject_id,most_likely_tool,x,y,p(most_likely_tool),p(true_positive),num_users"
            row = str(subject_id)+","

            # extract the most likely tool for this particular marking and convert it to
            # a string label
            tool_classification = cluster["tool_classification"][0].items()
            most_likely_tool,tool_probability = max(tool_classification, key = lambda x:x[1])
            tool_str = self.instructions[workflow_id][task_id]["tools"][int(most_likely_tool)]["marking tool"]
            tool_str = self.__csv_string__(tool_str)
            row += tool_str + "," + str(cluster["center"][0][0]) + "," + str(cluster["center"][0][1]) + "," + str(cluster["center"][1][0]) + "," + str(cluster["center"][1][1])
            # get the central coordinates next

            self.csv_files[key].write(row+"\n")



    def __shape_summary_output__(self,workflow_id,task_id,subject_id,aggregations,given_shape):
        """
        for a given shape, print out a summary of the all corresponding clusters  - one line more subject
        each line contains a count of the the number of such clusters which at least half the people marked
        the mean and median % of people to mark each cluster and the mean and median vote % for the
        most likely tool for each cluster. These last 4 values will help determine which subjects are "hard"
        :param workflow_id:
        :param task_id:
        :param subject_id:
        :param aggregations:
        :param shape:
        :return:
        """
        key = task_id + given_shape + "_summary"
        all_exist_probability = []
        all_tool_prob = []

        # start by figuring all the points which correspond to the desired type
        cluster_count = {}
        for tool_id in sorted(self.instructions[workflow_id][task_id]["tools"].keys()):
            tool_id = int(tool_id)

            assert task_id in self.workflows[workflow_id][1]
            shape = self.workflows[workflow_id][1][task_id][tool_id]
            if shape == given_shape:
                cluster_count[tool_id] = 0

        # now go through the actual clusters and count all which at least half of everyone has marked
        # or p(existence) >= 0.5 which is basically the same thing unless you've used weighted voting, IBCC etc.
        for cluster_index,cluster in aggregations[given_shape + " clusters"].items():
            if cluster_index == "all_users":
                continue

            prob_true_positive = cluster["existence"][0]["1"]
            if prob_true_positive > 0.5:
                tool_classification = cluster["tool_classification"][0].items()
                most_likely_tool,tool_prob = max(tool_classification, key = lambda x:x[1])
                all_tool_prob.append(tool_prob)
                cluster_count[int(most_likely_tool)] += 1

            # keep track of this no matter what the value is
            all_exist_probability.append(prob_true_positive)

        row = str(subject_id) + ","
        for tool_id in sorted(cluster_count.keys()):
            row += str(cluster_count[tool_id]) + ","

        # if there were no clusters found (at least which met the threshold) use empty columns
        if all_exist_probability == []:
            row += ",,"
        else:
            row += str(numpy.mean(all_exist_probability)) + "," + str(numpy.median(all_exist_probability)) + ","

        if all_tool_prob == []:
            row += ","
        else:
            row += str(numpy.mean(all_tool_prob)) + "," + str(numpy.median(all_tool_prob))

        self.csv_files[key].write(row+"\n")

    def __marking_header_setup__(self,workflow_id,task,tools,output_directory):
        """
        tools - says what sorts of different types of shapes/tools we have to do deal with for this task
        we can either give the output for each tool in a completely different csv file - more files, might
        be slightly overwhelming, but then we could make the column headers more understandable
        """
        fname = str(task)+self.instructions[workflow_id][task]["instruction"][:50]
        fname = self.__csv_string__(fname)

        if "polygon" in tools:
            key = task+"polygon_summary"
            self.csv_files[key] = open(output_directory+fname+"_polygons_summary.csv","wb")
            header = "subject_id,num_users,minimum_users_per_cluster,area(noise),tool_certainity"
            for tool_id in sorted(self.instructions[workflow_id][task]["tools"].keys()):
                tool = self.instructions[workflow_id][task]["tools"][tool_id]["marking tool"]
                tool = re.sub(" ","_",tool)
                header += ",area("+tool+")"
            self.csv_files[key].write(header+"\n")

            key = task+"polygon_heatmap"
            self.csv_files[key] = open(output_directory+fname+"_polygons_heatmap.csv","wb")
            header = "subject_id,num_users,pts"
            self.csv_files[key].write(header+"\n")

        if "point" in tools:
            key = task + "point"
            self.csv_files[key] = open(output_directory+fname+"_point.csv","wb")
            header = "subject_id,most_likely_tool,x,y,p(most_likely_tool),p(true_positive),num_users"
            self.csv_files[key].write(header+"\n")

            self.__summary_header_setup__(output_directory,fname,workflow_id,task,"point")

        if "line" in tools:
            key = task + "line"
            self.csv_files[key] = open(output_directory+fname+"_line.csv","wb")
            header = "subject_id,most_likely_tool,x1,y1,x2,y2,p(most_likely_tool),p(true_positive),num_users"
            self.csv_files[key].write(header+"\n")
            self.__summary_header_setup__(output_directory,fname,workflow_id,task,"line")

        if "rectangle" in tools:
            key = task + "rectangle"
            self.csv_files[key] = open(output_directory+fname+"_rectangle.csv","wb")
            header = "subject_id,most_likely_tool,x1,y1,x2,y2,num_users"
            self.csv_files[key].write(header+"\n")
            self.__summary_header_setup__(output_directory,fname,workflow_id,task,"rectangle")


    def __summary_header_setup__(self,output_directory,fname,workflow_id,task,shape):
        """
        all shape aggregation will have a summary file - with one line per subject
        :return:
        """
        # the summary file will contain just line per subject
        key = task + shape +"_summary"
        self.csv_files[key] = open(output_directory+fname+"_"+shape+"_summary.csv","wb")
        header = "subject_id"
        # extract only the tools which can actually make point markings
        for tool_id in sorted(self.instructions[workflow_id][task]["tools"].keys()):
            tool_id = int(tool_id)
            # self.workflows[workflow_id][0] is the list of classification tasks
            # we want [1] which is the list of marking tasks
            assert task in self.workflows[workflow_id][1]
            found_shape = self.workflows[workflow_id][1][task][tool_id]
            if found_shape == shape:
                tool_label = self.instructions[workflow_id][task]["tools"][tool_id]["marking tool"]
                tool_label = self.__csv_string__(tool_label)
                header += "," + tool_label
        header += ",mean_probability,median_probability,mean_tool,median_tool"
        self.csv_files[key].write(header+"\n")

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
            self.csv_files[key].write(row+"\n")

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
        self.csv_files[key].write(row+"\n")