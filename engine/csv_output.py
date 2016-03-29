from __future__ import print_function
import os
import numpy
import tarfile
import math
import sys
# for sphinx documentation, there seems to be trouble with importing shapely
# so for the time being, if we can't import it, since it doesn't actually matter
# for documentation, just have all the imported things wind up being undefined
try:
    import shapely.geometry as geometry
except OSError:
    pass
import helper_functions
import numpy as np
from helper_functions import warning

__author__ = 'greg'


class CsvOut:
    def __init__(self,project):
        # assert isinstance(project,aggregation_api.AggregationAPI)
        self.project = project

        self.project_id = project.project_id
        self.instructions = project.instructions
        self.workflow_names = project.workflow_names
        self.workflows = project.workflows

        print("workflows are " + str(self.workflows))

        self.__yield_aggregations__ = project.__yield_aggregations__
        self.retirement_thresholds = project.retirement_thresholds
        self.versions = project.versions

        self.__count_subjects_classified__ = project.__count_subjects_classified__

        # dictionary to hold the output files
        self.csv_files = {}
        # stores the file names
        self.file_names = {}
        self.workflow_directories = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __classification_output__(self,workflow_id,task_id,subject_id,aggregations,shape_id=None,followup_id=None):
        """
        add a row to both the summary and detailed csv output files
        """
        # a dictionary containing the index id of each answer and its corresponding label
        answer_dict = self.instructions[workflow_id][task_id]["answers"]

        # start with the summary file
        id_ = (workflow_id,task_id,shape_id,followup_id,"summary")

        try:
            self.__add_summary_row__(id_,subject_id,aggregations,answer_dict)

            id_ = (workflow_id,task_id,shape_id,followup_id,"detailed")
            self.__add_detailed_row__(id_,subject_id,aggregations,answer_dict)
        except ValueError:
             warning("empty aggregations for workflow id " + str(workflow_id) + " task id " + str(task_id) + " and subject id" + str(subject_id) + " -- skipping")

    def __add_detailed_row__(self,id_,subject_id,results,answer_dict):
        """
        given the results for a given workflow/task and subject_id (and possibly shape and follow up id for marking)
        give a detailed results with the probabilities for each class
        :param id_:
        :param subject_id:
        :param results:
        :param answer_dict:
        :return:
        """
        votes,num_users = results

        with open(self.file_names[id_],"a") as results_file:
            results_file.write(str(subject_id))

            for answer_key in sorted(answer_dict.keys()):
                # if no one chose this particular answer, the probability was 0
                if str(answer_key) not in votes:
                    percentage = 0
                else:
                    percentage = votes[str(answer_key)]

                results_file.write(","+str(percentage))

            results_file.write(","+str(num_users)+"\n")

    def __add_summary_row__(self,id_,subject_id,results,answer_dict):
        """
        given a result for a specific subject (and possibily a specific cluster within that specific subject)
        add one row of results to the summary file. that row contains
        subject_id,tool_index,cluster_index,most_likely,p(most_likely),shannon_entropy,mean_agreement,median_agreement,num_users
        tool_index & cluster_index are only there if we have a follow up to marking task
        :param id_:
        :param subject_id:
        :param results:
        :param answer_dict:
        :return:
        """
        votes,num_users = results

        # get the top choice
        try:
            most_likely,top_probability = max(votes.items(), key = lambda x:x[1])
        except ValueError:
            warning(results)
            raise

        # extract the text corresponding to the most likely answer
        most_likely_label = answer_dict[int(most_likely)]
        # and get rid of any bad characters
        most_likely_label = helper_functions.csv_string(most_likely_label)

        probabilities = votes.values()
        entropy = self.__shannon_entropy__(probabilities)

        mean_p = np.mean(votes.values())
        median_p = np.median(votes.values())

        with open(self.file_names[id_],"a") as results_file:
            results_file.write(str(subject_id)+",")

            # write out details regarding the top choice
            # this might not be a useful value if multiple choices are allowed - in which case just ignore it
            results_file.write(str(most_likely_label)+","+str(top_probability))
            # write out some summaries about the distributions of people's answers
            # again entropy probably only makes sense if only one answer is allowed
            # and mean_p and median_p probably only make sense if multiple answers are allowed
            # so people will need to pick and choose what they want
            results_file.write(","+str(entropy)+","+str(mean_p)+","+str(median_p))
            # finally - how many people have seen this subject for this task
            results_file.write(","+str(num_users)+"\n")

    def __get_filename__(self,workflow_id,task_id,summary=False,tool_id=None,followup_id=None):
        """
        use the user's instructions to help create a file name to store the results in
        :param workflow_id:
        :param task_id:
        :param summary:
        :return:
        """
        assert (tool_id is None) or (followup_id is not None)

        # read in the instructions
        # if just a simple classification question
        if tool_id is None:
            instructions = self.instructions[workflow_id][task_id]["instruction"]
        # else a follow up question to a marking - so the instructions are stored in a sligghtly different spot
        else:
            instructions = self.instructions[workflow_id][task_id]["tools"][tool_id]["followup_questions"][followup_id]["question"]

        fname = str(task_id) + instructions[:50]
        if summary:
            fname += "_summary"
        # get rid of any characters (like extra ","s) that could cause problems
        fname = helper_functions.csv_string(fname)
        fname += ".csv"

        return fname

    def __classification_header__(self,output_directory,workflow_id,task_id,tool_id=None,followup_id=None):
        assert (tool_id is None) or (followup_id is not None)
        # start with the detailed results
        fname = self.__get_filename__(workflow_id,task_id,tool_id=tool_id,followup_id=followup_id)

        id_ = (workflow_id,task_id,tool_id,followup_id,"detailed")
        self.file_names[id_] = output_directory+fname
        with open(output_directory+fname,"wb") as detailed_results:
            # now write the headers
            detailed_results.write("subject_id")

            if tool_id is not None:
                detailed_results.write(",cluster_id")

            answer_dict = self.instructions[workflow_id][task_id]["answers"]
            for answer_key in sorted(answer_dict.keys()):
                # break this up into multiple lines so we can be sure that the answers are sorted correctly
                # order might not matter in the end, but just to be sure
                answer = answer_dict[answer_key]
                answer_string = helper_functions.csv_string(answer)[:50]
                detailed_results.write(",p("+answer_string+")")

            detailed_results.write(",num_users\n")

        # now setup the summary file
        fname = self.__get_filename__(workflow_id,task_id,summary = True,tool_id=tool_id,followup_id=followup_id)
        id_ = (workflow_id,task_id,tool_id,followup_id,"summary")
        self.file_names[id_] = output_directory+fname
        with open(output_directory+fname,"wb") as summary_results:
            self.csv_files[id_] = summary_results
            self.csv_files[id_].write("subject_id,")

            if tool_id is not None:
                self.csv_files[id_].write("cluster_id,")

            self.csv_files[id_].write("most_likely,p(most_likely),shannon_entropy,mean_agreement,median_agreement,num_users\n")

    def __make_files__(self,workflow_id):
        """
        create all of the files necessary for this workflow
        :param workflow_id:
        :return:
        """
        # close any previously used files (and delete their pointers)
        for f in self.csv_files.values():
            f.close()
        self.csv_files = {}

        # now create a sub directory specific to the workflow
        try:
            workflow_name = self.workflow_names[workflow_id]
        except KeyError:
            warning(self.workflows)
            warning(self.workflow_names)
            raise

        workflow_name = helper_functions.csv_string(workflow_name)
        output_directory = "/tmp/"+str(self.project_id)+"/" +str(workflow_id) + "_" + workflow_name + "/"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.workflow_directories[workflow_id] = output_directory

        classification_tasks,marking_tasks,survey_tasks = self.workflows[workflow_id]

        # go through the classification tasks - they will either be simple c. tasks (one answer allowed)
        # multiple c. tasks (more than one answer allowed) and possibly a follow up question to a marking
        for task_id in classification_tasks:
            # is this task a simple classification task?
            # don't care if the questions allows for multiple answers, or requires a single one
            if classification_tasks[task_id] in ["single","multiple"]:
                self.__classification_header__(output_directory,workflow_id,task_id)

            else:
                # this classification task is actually a follow up to a marking task
                for tool_id in classification_tasks[task_id]:
                    for followup_id,answer_type in enumerate(classification_tasks[task_id][tool_id]):
                        # instructions = self.instructions[workflow_id][task_id]["tools"][tool_id]["followup_questions"][followup_index]["question"]
                        self.__classification_header__(output_directory,workflow_id,task_id,tool_id,followup_id)
                        # id_ = (task_id,tool_id,followup_index)
                        # if answer_type == "single":
                        #     self.__single_response_csv_header__(output_directory,id_,instructions)
                        # else:
                        #     self.__multi_response_csv_header__(output_directory,id_,instructions)

        # now set things up for the marking tasks
        for task_id in marking_tasks:
            shapes = set(marking_tasks[task_id])
            self.__marking_header_setup__(workflow_id,task_id,shapes,output_directory)

        # and finally the survey tasks
        for task_id in survey_tasks:
            instructions = self.instructions[workflow_id][task_id]
            self.__survey_header_setup__(output_directory,task_id,instructions)

        return output_directory

    def __shannon_entropy__(self,probabilities):
        return -sum([p*math.log(p) for p in probabilities])


    def __single_choice_classification_row__(self,answers,task_id,subject_id,results,cluster_index=None):
        """
        output a row for a classification task which only allowed allowed one answer
        global_task_id => the task might actually be a subtask, in which case the id needs to contain
        the task id, tool and follow up question id
        :param global_task_id:
        :param subject_id:
        :param results:
        :return:
        """
        # since only one choice is allowed, go for the maximum
        votes,num_users = results
        if votes == {}:
            return
        most_likely,top_probability = max(votes.items(), key = lambda x:x[1])

        # extract the text corresponding to the most likely answer
        most_likely_label = answers[int(most_likely)]
        # this corresponds to when the question is a follow up
        if isinstance(most_likely_label,dict):
            most_likely_label = most_likely_label["label"]
        most_likely_label = helper_functions.csv_string(most_likely_label)

        probabilities = votes.values()
        entropy = self.__shannon_entropy__(probabilities)

        row = str(subject_id)+","
        if cluster_index is not None:
            row += str(cluster_index) + ","
        row += most_likely_label+","+str(top_probability)+","+str(entropy)+","+str(num_users)+"\n"

        # finally write the stuff out to file
        self.csv_files[task_id].write(row)

    def __subject_output__(self,workflow_id,subject_id,aggregations):
        """
        add csv rows for all the output related to this particular workflow/subject_id
        :param workflow_id:
        :param subject_id:
        :param aggregations:
        :return:
        """

        classification_tasks,marking_tasks,survey_tasks = self.workflows[workflow_id]

        for task_id,task_type in classification_tasks.items():
            # a subject might not have results for all tasks
            if task_id not in aggregations:
                continue

            # we have follow up questions
            if isinstance(task_type,dict):
                for tool_id in task_type:
                    for followup_index,answer_type in enumerate(task_type[tool_id]):
                        # what sort of shape are we looking for - help us find relevant clusters
                        shape = self.workflows[workflow_id][1][task_id][tool_id]
                        for cluster_index,cluster in aggregations[task_id][shape + " clusters"].items():
                            if cluster_index == "all_users":
                                continue

                            classification = cluster["tool_classification"][0]
                            most_likely_tool,_ = max(classification.items(),key = lambda x:x[1])

                            # only consider clusters which most likely correspond to the correct tool
                            if int(most_likely_tool) != int(tool_id):
                                continue

                            possible_answers = self.instructions[workflow_id][task_id]["tools"][tool_id]["followup_questions"][followup_index]["answers"]
                            if "followup_question" not in aggregations[task_id][shape + " clusters"][cluster_index]:
                                print("missing follow up response")
                                continue

                            try:
                                results = aggregations[task_id][shape + " clusters"][cluster_index]["followup_question"][str(followup_index)]
                            except KeyError:
                                warning(aggregations[task_id][shape + " clusters"][cluster_index])
                                raise
                            id_ = task_id,tool_id,followup_index
                            if answer_type == "single":
                                self.__single_choice_classification_row__(possible_answers,id_,subject_id,results,cluster_index)
                            else:
                                self.__multi_choice_classification_row__(possible_answers,id_,subject_id,results,cluster_index)
            else:
                results = aggregations[task_id]
                self.__classification_output__(workflow_id,task_id,subject_id,results)

        for task_id,possible_shapes in marking_tasks.items():
            for shape in set(possible_shapes):
                # not every task have been done for every aggregation
                if task_id in aggregations:
                    if shape == "polygon":
                        self.__polygon_row__(workflow_id,task_id,subject_id,aggregations[task_id])
                        self.__polygon_summary_output__(workflow_id,task_id,subject_id,aggregations[task_id])
                    else:
                        self.__marking_row__(workflow_id,task_id,subject_id,aggregations[task_id],shape)
                        self.__shape_summary_output__(workflow_id,task_id,subject_id,aggregations,shape)

        for task_id in survey_tasks:
            instructions = self.instructions[workflow_id][task_id]

            # id_ = (task_id,"summary")
            # with open(self.file_names[id_],"a") as f:
            #     summary_line = self.__survey_summary_row(aggregations)
            #     f.write(str(subject_id)+summary_line)

            id_ = (task_id,"detailed")
            with open(self.file_names[id_],"a") as f:
                detailed_lines = self.__survey_row__(instructions,aggregations)
                for l in detailed_lines:
                    if subject_id == 873807:
                        print(l)
                    f.write(str(subject_id)+l)

    def __survey_header_setup__(self,output_directory,task_id,instructions):
        """
        create the csv output file for a survey task
        and give the header row
        :param output_directory:
        :param task_id:
        :param instructions:
        :return:
        """
        # # start with the summary files
        # fname = output_directory+str(task_id) + "_survey_summary.csv"
        # self.file_names[(task_id,"summary")] = fname
        # with open(fname,"wb") as f:
        #     f.write("subject_id,pielou_index\n")

        # and then the detailed files
        fname = output_directory+str(task_id) + "_survey_detailed.csv"
        self.file_names[(task_id,"detailed")] = fname

        # now write the header
        header = "subject_id,num_classifications,pielou_score,species,number_of_votes_for_species"

        # todo - we'll assume, for now, that "how many" is always the first question
        for followup_id in instructions["questionsOrder"]:
            multiple_answers = instructions["questions"][followup_id]["multiple"]
            label = instructions["questions"][followup_id]["label"]

            # the question "how many" is treated differently - we'll give the minimum, maximum and mostly likely
            if followup_id == "HWMN":
                header += ",minimum_number_of_animals,most_likely_number_of_animals,percentage,maximum_number_of_animals"
            else:
                if "behavior" in label:
                    stem = "behaviour:"
                elif "behaviour" in label:
                    stem = "behaviour:"
                else:
                    stem = helper_functions.csv_string(label)

                if not multiple_answers:
                    header += ",most_likely(" + stem + ")"

                for answer_id in instructions["questions"][followup_id]["answersOrder"]:
                    header += ",percentage(" + stem + helper_functions.csv_string(instructions["questions"][followup_id]["answers"][answer_id]["label"]) +")"

        with open(fname,"wb") as f:
            f.write(header+"\n")



    def __survey_how_many__(self,instructions,aggregations,species_id):
        """
        return the columns for the question how many animals are present in an image
        for survey tasks
        keep in mind that counts can be buckets - i.e. 10-19
        columns:
        min - the minimum number of animals anyone said
        most_likely - the bucket with the highest percentage
        percentage - how many people said the most likely
        max - the maximum number of animals anyone said
        :return:
        """
        followup_id = "HWMN"
        followup_question = instructions["questions"][followup_id]
        votes = aggregations[species_id]["followup"][followup_id].items()
        # sort by num voters
        sorted_votes = sorted(votes,key = lambda x:x[1],reverse=True)
        candidates,vote_counts = zip(*sorted_votes)
        candidates = list(candidates)

        # top candidate is the most common response to the question of how many animals there are in the subject
        top_candidate = followup_question["answers"][candidates[0]]["label"]
        percentage = vote_counts[0]/float(sum(vote_counts))

        # what is the minimum/maximum number of animals of this species that people said were in the subject?
        answer_order = followup_question["answersOrder"]
        # resort by position in answer order
        candidates.sort(key = lambda x:answer_order.index(x))
        minimum_species = followup_question["answers"][candidates[0]]["label"]
        maximum_species = followup_question["answers"][candidates[-1]]["label"]

        return "," + str(minimum_species) + "," + str(top_candidate) + "," + str(percentage) + "," + str(maximum_species)

    # def __get_species_in_subject(self,aggregations):
    #     """
    #     use Ali's and Margaret's code to determine how many species are a given subject
    #     and return those X top species
    #     :return:
    #     """
    #     print(aggregations)
    #     num_species = int(np.median(aggregations["num species"]))
    #     assert(num_species >= 1)
    #     # sort the species by the number of votes
    #     species_by_vote = []
    #
    #     for species_id in aggregations:
    #         if species_id not in ["num users","num species",""]:
    #             species_by_vote.append((species_id,aggregations[species_id]["num votes"]))
    #     sorted_species = sorted(species_by_vote,key = lambda x:x[1],reverse=True)
    #
    #     return sorted_species[:num_species]


    def __survey_row__(self,instructions,aggregations):
        """
        for a given workflow, task and subject print one row of aggregations per species found to a csv file
        where the task correspond to a survey task
        :param workflow_id:
        :param task_id:
        :param subject_id:
        :param aggregations:
        :return:
        """
        # what we are returning (to be printed out to file elsewhere)
        rows = []

        # in dev - for a small project a few bad aggregations got into the system - so filer them out
        if max(aggregations["num species"]) == 0:
            return []

        # on average, how many species did people see?
        # note - nothing here (or empty or what ever) counts as a species - we just won't give any follow up
        # answer responses
        species_in_subject = aggregations["num species in image"]

        views_of_subject = aggregations["num users"]

        pielou = aggregations["pielou index"]

        # only go through the top X species - where X is the median number of species seen
        for species_id,_ in species_in_subject:
            if species_id == "num users":
                continue

            # how many people voted for this species?
            num_votes = aggregations[species_id]["num votes"]
            percentage = num_votes/float(views_of_subject)

            # extract the species name - just to be sure, make sure that the label is "csv safe"
            species_label = helper_functions.csv_string(instructions["species"][species_id])
            row = "," + str(views_of_subject) + "," + str(pielou) + "," + species_label + "," +str(percentage) + "," + str(num_votes)

            # if there is nothing here - there are no follow up questions so just move on
            # same with FR - fire, NTHNG - nothing
            if species_id in ["NTHNGHR","NTHNG","FR"]:
                break

            # do the how many question first
            row += self.__survey_how_many__(instructions,aggregations,species_id)

            # now go through each of the other follow up questions
            for followup_id in instructions["questionsOrder"]:
                followup_question = instructions["questions"][followup_id]

                if followup_question["label"] == "How many?":
                    # this gets dealt with separately
                    continue

                # this follow up question might not be relevant to the particular species
                if followup_id not in aggregations[species_id]["followup"]:
                    for answer_id in instructions["questions"][followup_id]["answersOrder"]:
                        row += ","
                else:
                    votes = aggregations[species_id]["followup"][followup_id]

                    # if users are only allowed to pick a single answer - return the most likely answer
                    # but still give the individual break downs
                    multiple_answers = instructions["questions"][followup_id]["multiple"]
                    if not multiple_answers:
                        votes = aggregations[species_id]["followup"][followup_id].items()
                        answers =(instructions["questions"][followup_id]["answers"])
                        top_candidate,percent = self.__get_top_survey_followup__(votes,answers)

                        row += "," + str(top_candidate) + "," + str(percent)

                    for answer_id in instructions["questions"][followup_id]["answersOrder"]:
                        if answer_id in votes:
                            row += "," + str(votes[answer_id]/float(num_votes))
                        else:
                            row += ",0"

            rows.append(row+"\n")

        return rows

    def __get_top_survey_followup__(self,votes,answers):
        """
        for a particular follow up classification question in a survey task where only one answer is allowed
        return the top/most likely classification and its associated probability
        :param aggregations:
        :return:
        """
        # list answer in decreasing order
        sorted_votes = sorted(votes,key = lambda x:x[1],reverse=True)
        candidates,vote_counts = zip(*sorted_votes)

        top_candidate = candidates[0]
        percent = vote_counts[0]/float(sum(vote_counts))

        return answers[top_candidate]["label"],percent

    def __polygon_row__(self,workflow_id,task_id,subject_id,aggregations):
        id_ = task_id,"polygon","detailed"

        # for p_index,cluster in aggregations["polygon clusters"].items():
        #     if p_index == "all_users":
        #         continue
        #
        #     tool_classification = cluster["tool_classification"][0].items()
        #     most_likely_tool,tool_probability = max(tool_classification, key = lambda x:x[1])
        #     total_area[int(most_likely_tool)] += cluster["area"]

        for p_index,cluster in aggregations["polygon clusters"].items():
            if p_index == "all_users":
                continue

            tool_classification = cluster["tool_classification"][0].items()
            most_likely_tool,tool_probability = max(tool_classification, key = lambda x:x[1])
            tool = self.instructions[workflow_id][task_id]["tools"][int(most_likely_tool)]["marking tool"]
            tool = helper_functions.csv_string(tool)

            for polygon in cluster["center"]:
                p = geometry.Polygon(polygon)

                row = str(subject_id) + ","+ str(p_index)+ ","+ tool + ","+ str(p.area/float(cluster["image area"])) + ",\"" +str(polygon) + "\""
                self.csv_files[id_].write(row+"\n")

    def __write_out__(self,subject_set = None,compress=True):
        """
        create the csv outputs for a given set of workflows
        the workflows are specified by self.workflows which is determined when the aggregation engine starts
        a zipped file is created in the end
        """
        assert (subject_set is None) or isinstance(subject_set,set)

        project_prefix = str(self.project_id)

        # with open("/tmp/"+str(self.project_id)+"/readme.md", "w") as readme_file:
        #     readme_file.truncate()

        if not os.path.exists("/tmp/"+str(self.project_id)):
            os.makedirs("/tmp/"+str(self.project_id))

        # go through each workflow indepedently
        for workflow_id in self.workflows:
            print("writing out workflow " + str(workflow_id))

            if self.__count_subjects_classified__(workflow_id) == 0:
                print("skipping due to no subjects being classified for the given workflow")
                continue

            # # create the output files for this workflow
            self.__make_files__(workflow_id)

            # results are going to be ordered by subject id (because that's how the results are stored)
            # so we can going to be cycling through task_ids. That's why we can't loop through classification_tasks etc.
            for subject_id,aggregations in self.__yield_aggregations__(workflow_id,subject_set):
                self.__subject_output__(workflow_id,subject_id,aggregations)

        for f in self.csv_files.values():
            f.close()

        # # add some final details to the read me file

        try:
            with open("/tmp/"+project_prefix+"/readme.md", "w") as readme_file:
                # readme_file.write("Details and food for thought:\n")
                with open("/app/engine/readme.txt","rb") as f:
                    text = f.readlines()
                    for l in text:
                        readme_file.write(l)
        except IOError as e:


            with open("/tmp/"+project_prefix+"/readme.md", "w") as readme_file:
                readme_file.write("There was an IO error - \n")
                readme_file.write(str(e) + "\n")
                readme_file.write(os.getcwd())
            #     readme_file.write("There are no retired subjects for this project")

        if compress:
            tar_file_path = "/tmp/" + project_prefix + "_export.tar.gz"
            with tarfile.open(tar_file_path, "w:gz") as tar:
                tar.add("/tmp/"+project_prefix+"/")

            return tar_file_path



    def __marking_row__(self,workflow_id,task_id,subject_id,aggregations,shape):
        """
        output for line segments
        :param workflow_id:
        :param task_id:
        :param subject_id:
        :param aggregations:
        :return:
        """
        key = task_id,shape,"detailed"
        for cluster_index,cluster in aggregations[shape + " clusters"].items():
            if cluster_index == "all_users":
                continue

            # build up the row bit by bit to have the following structure
            # "subject_id,most_likely_tool,x,y,p(most_likely_tool),p(true_positive),num_users"
            row = str(subject_id)+","
            # todo for now - always give the cluster index
            row += str(cluster_index)+","

            # extract the most likely tool for this particular marking and convert it to
            # a string label
            try:
                tool_classification = cluster["tool_classification"][0].items()
            except KeyError:
                warning(shape)
                warning(cluster)
                raise
            most_likely_tool,tool_probability = max(tool_classification, key = lambda x:x[1])
            tool_str = self.instructions[workflow_id][task_id]["tools"][int(most_likely_tool)]["marking tool"]
            row += helper_functions.csv_string(tool_str) + ","

            # get the central coordinates next
            for center_param in cluster["center"]:
                if isinstance(center_param,list) or isinstance(center_param,tuple):
                    row += "\"" + str(tuple(center_param)) + "\","
                else:
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
        relevant_tools = [tool_id for tool_id,tool_shape in enumerate(self.workflows[workflow_id][1][task_id]) if tool_shape == given_shape]
        counter = {t:{} for t in relevant_tools}
        aggreement = []

        prob_true_positive = []#{t:[] for t in relevant_tools}

        for cluster_index,cluster in aggregations[task_id][given_shape + " clusters"].items():
            if cluster_index == "all_users":
                continue

            # how much agreement was their on the most likely tool?
            tool_classification = cluster["tool_classification"][0].items()
            most_likely_tool,tool_prob = max(tool_classification, key = lambda x:x[1])
            aggreement.append(tool_prob)

            prob_true_positive.append(cluster["existence"][0]["1"])

            for u,t in zip(cluster["users"],cluster["tools"]):
                if u in counter[t]:
                    counter[t][u] += 1
                else:
                    counter[t][u] = 1

            # print


        # # start by figuring all the points which correspond to the desired type
        # cluster_count = {}
        # for tool_id in sorted(self.instructions[workflow_id][task_id]["tools"].keys()):
        #     tool_id = int(tool_id)
        #
        #     assert task_id in self.workflows[workflow_id][1]
        #     shape = self.workflows[workflow_id][1][task_id][tool_id]
        #     if shape == given_shape:
        #         cluster_count[tool_id] = 0
        #
        # # now go through the actual clusters and count all which at least half of everyone has marked
        # # or p(existence) >= 0.5 which is basically the same thing unless you've used weighted voting, IBCC etc.
        # for cluster_index,cluster in aggregations[task_id][given_shape + " clusters"].items():
        #     if cluster_index == "all_users":
        #         continue
        #
        #     prob_true_positive = cluster["existence"][0]["1"]
        #     if prob_true_positive > 0.5:
        #         tool_classification = cluster["tool_classification"][0].items()
        #         most_likely_tool,tool_prob = max(tool_classification, key = lambda x:x[1])
        #         all_tool_prob.append(tool_prob)
        #         cluster_count[int(most_likely_tool)] += 1
        #
        #     # keep track of this no matter what the value is
        #     all_exist_probability.append(prob_true_positive)

        row = str(subject_id) + ","
        for tool_id in sorted(counter.keys()):
            tool_count = counter[tool_id].values()
            if tool_count == []:
                row += "0,"
            else:
                row += str(numpy.median(tool_count)) + ","

        if prob_true_positive == []:
            row += "NA,NA,"
        else:
            row += str(numpy.mean(prob_true_positive)) + "," + str(numpy.median(prob_true_positive)) + ","

        if aggreement == []:
            row += "NA,NA"
        else:

            row += str(numpy.mean(aggreement)) + "," + str(numpy.median(aggreement))



        # # if there were no clusters found (at least which met the threshold) use empty columns
        # if all_exist_probability == []:
        #     row += ",,"
        # else:
        #     row += str(numpy.mean(all_exist_probability)) + "," + str(numpy.median(all_exist_probability)) + ","
        #
        # if all_tool_prob == []:
        #     row += ","
        # else:
        #     row += str(numpy.mean(all_tool_prob)) + "," + str(numpy.median(all_tool_prob))
        #
        id_ = task_id,given_shape,"summary"
        self.csv_files[id_].write(row+"\n")

    def __marking_header_setup__(self,workflow_id,task_id,shapes,output_directory):
        """
        - create the csv output files for each workflow/task pairing where the task is a marking
        also write out the header line
        - since different tools (for the same task) can have completely different shapes, these shapes should
        be printed out to different files - hence the multiple output files
        - we will give both a summary file and a detailed report file
        """
        for shape in shapes:
            fname = str(task_id) + self.instructions[workflow_id][task_id]["instruction"][:50]
            fname = helper_functions.csv_string(fname)
            # fname += ".csv"


            self.file_names[(task_id,shape,"detailed")] = fname + "_" + shape + ".csv"
            self.file_names[(task_id,shape,"summary")] = fname + "_" + shape + "_summary.csv"

            # polygons - since they have an arbitary number of points are handled slightly differently
            if shape == "polygon":
                id_ = task_id,shape,"detailed"
                self.csv_files[id_] = open(output_directory+fname+"_"+shape+".csv","wb")
                self.csv_files[id_].write("subject_id,cluster_index,most_likely_tool,area,list_of_xy_polygon_coordinates\n")

                id_ = task_id,shape,"summary"
                self.csv_files[id_] = open(output_directory+fname+"_"+shape+"_summary.csv","wb")
                # self.csv_files[id_].write("subject_id,\n")
                polygon_tools = [t_index for t_index,t in enumerate(self.workflows[workflow_id][1][task_id]) if t == "polygon"]
                header = "subject_id,"
                for tool_id in polygon_tools:
                    tool = self.instructions[workflow_id][task_id]["tools"][tool_id]["marking tool"]
                    tool = helper_functions.csv_string(tool)
                    header += "area("+tool+"),"
                self.csv_files[id_].write(header+"\n")

            else:
                id_ = task_id,shape,"detailed"
                # fname += "_"+shape+".csv"
                self.csv_files[id_] = open(output_directory+fname+"_"+shape+".csv","wb")

                header = "subject_id,cluster_index,most_likely_tool,"
                if shape == "point":
                    header += "x,y,"
                elif shape == "rectangle":
                    # todo - fix this
                    header += "x1,y1,x2,y2,"
                elif shape == "line":
                    header += "x1,y1,x2,y2,"
                elif shape == "ellipse":
                    header += "x1,y1,r1,r2,theta,"

                header += "p(most_likely_tool),p(true_positive),num_users"
                self.csv_files[id_].write(header+"\n")
                # do the summary output else where
                self.__summary_header_setup__(output_directory,workflow_id,fname,task_id,shape)

    def __summary_header_setup__(self,output_directory,workflow_id,fname,task_id,shape):
        """
        all shape aggregation will have a summary file - with one line per subject
        :return:
        """
        # the summary file will contain just line per subject
        id_ = task_id,shape,"summary"
        self.csv_files[id_] = open(output_directory+fname+"_"+shape+"_summary.csv","wb")
        header = "subject_id"
        # extract only the tools which can actually make point markings
        for tool_id in sorted(self.instructions[workflow_id][task_id]["tools"].keys()):
            tool_id = int(tool_id)
            # self.workflows[workflow_id][0] is the list of classification tasks
            # we want [1] which is the list of marking tasks
            found_shape = self.workflows[workflow_id][1][task_id][tool_id]
            if found_shape == shape:
                tool_label = self.instructions[workflow_id][task_id]["tools"][tool_id]["marking tool"]
                tool_label = helper_functions.csv_string(tool_label)
                header += ",median(" + tool_label +")"
        header += ",mean_probability,median_probability,mean_tool,median_tool"
        self.csv_files[id_].write(header+"\n")

    # def __polygon_heatmap_output__(self,workflow_id,task_id,subject_id,aggregations):
    #     """
    #     print out regions according to how many users selected that user - so we can a heatmap
    #     of the results
    #     :param workflow_id:
    #     :param task_id:
    #     :param subject_id:
    #     :param aggregations:
    #     :return:
    #     """
    #     key = task_id+"polygon_heatmap"
    #     for cluster_index,cluster in aggregations["polygon clusters"].items():
    #         # each cluster refers to a specific tool type - so there can actually be multiple blobs
    #         # (or clusters) per cluster
    #         # not actually clusters
    #
    #         if cluster_index in ["param","all_users"]:
    #             continue
    #
    #         if cluster["tool classification"] is not None:
    #             # this result is not relevant to the heatmap
    #             continue
    #
    #         row = str(subject_id) + "," + str(cluster["num users"]) + ",\"" + str(cluster["center"]) + "\""
    #         self.csv_files[key].write(row+"\n")

    def __polygon_summary_output__(self,workflow_id,task_id,subject_id,aggregations):
        """
        print out a csv summary of the polygon aggregations (so not the individual xy points)
        need to know the workflow and task id so we can look up the instructions
        that way we can know if there is no output for a given tool - that tool wouldn't appear
        at all in the aggregations
        """
        polygon_tools = [t_index for t_index,t in enumerate(self.workflows[workflow_id][1][task_id]) if t == "polygon"]

        total_area = {t:0 for t in polygon_tools}

        id_ = task_id,"polygon","summary"
        for p_index,cluster in aggregations["polygon clusters"].items():
            if p_index == "all_users":
                continue

            tool_classification = cluster["tool_classification"][0].items()
            most_likely_tool,tool_probability = max(tool_classification, key = lambda x:x[1])
            total_area[int(most_likely_tool)] += cluster["area"]

        row = str(subject_id)
        for t in sorted([int(t) for t in polygon_tools]):
            row += ","+ str(total_area[t])

        self.csv_files[id_].write(row+"\n")

if __name__ == "__main__":
    import aggregation_api
    project_id = sys.argv[1]
    project = aggregation_api.AggregationAPI(project_id,"development")

    w = CsvOut(project)
    w.__write_out__()