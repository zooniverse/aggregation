from __future__ import print_function
import json
from helper_functions import warning
import tarfile
import numpy as np
from abc import ABCMeta, abstractmethod
"""
sets up the output for transcription projects like Annotate and Shakespeare's world
before being emailed out
"""
__author__ = 'ggdhines'


class TranscriptionOutput:
    __metaclass__ = ABCMeta

    def __init__(self,project):
        self.project = project

        self.workflow_id = self.project.workflows.keys()[0]
        self.metadata = self.project.__get_subject_metadata__(self.workflow_id)

    def __json_output__(self):
        aggregations_to_json = dict()
        print("creating json output ready")
        # by using metadata.keys, we automatically restrict the results to retired subjects
        for count, (subject_id, aggregations) in enumerate(self.project.__yield_aggregations__(self.workflow_id,self.metadata.keys())):
            print(subject_id)
            if isinstance(aggregations, str):
                aggregations = json.loads(aggregations)
            try:
                aggregations_to_json[subject_id] = self.__subject_to_json__(subject_id,aggregations)
            except IndexError:
                print("skipping " + str(subject_id))

        json.dump(aggregations_to_json, open("/tmp/" + str(self.project.project_id) + ".json", "wb"))
        self.__tar_output__()

    @abstractmethod
    def __subject_to_json__(self,subject_id,aggregation):
        """
        transform the aggregation results into a json output for just one subject
        :param aggregation: the aggregation results for just one subject
        :return:
        """
        return dict()

    def __tar_output__(self):
        aws_tar = self.project.__get_aws_tar_name__()
        print("saving json results")
        with tarfile.open("/tmp/"+aws_tar,mode="w") as t:
            t.add("/tmp/"+str(self.project.project_id)+".json")




class ShakespearesWorldOutput(TranscriptionOutput):
    def __init__(self,project):
        TranscriptionOutput.__init__(self,project)

    def __subject_to_json__(self,subject_id,aggregation):
        """
        transform the aggregation results into a json output for just one subject
        :param aggregation:
        :return:
        """
        subject_json = {"text":[],"individual transcriptions":[], "accuracy":[], "coordinates" : [],"users_per_line":[]}
        subject_json["metadata"] = self.metadata[subject_id]
        subject_json["zooniverse subject id"] = subject_id

        clusters_by_line = {}


        for key,cluster in aggregation["T2"]["text clusters"].items():
            if key == "all_users":
                continue
            if isinstance(cluster,str):
                warning("cluster is in string format for some reason")
                cluster = json.loads(cluster)

            try:
                # for dev only since we may not have updated every transcription
                if cluster["cluster members"] == []:
                    continue
            except TypeError:
                warning(cluster)
                warning()
                raise

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
                # add in the line of text
                subject_json["text"].append(merged_line)

                # now add in the aligned individual transcriptions
                # use the first cluster we found for this line as a "representative cluster"
                rep_cluster = cluster_set[0]

                zooniverse_ids = []
                for user_id in rep_cluster["cluster members"]:
                    zooniverse_login_name = self.project.__get_login_name__(user_id)

                    # todo - not sure why None can be returned but does seem to happen
                    if zooniverse_login_name is not None:
                        # new_json[subject_id]["users_per_line"].append(zooniverse_login_name)
                        zooniverse_ids.append(zooniverse_login_name)
                    else:
                        zooniverse_ids.append("None")

                # todo - if a line is transcribed completely but in distinct separate parts
                # todo - this may cause trouble
                subject_json["individual transcriptions"].append(rep_cluster["aligned_text"])
                subject_json["users_per_line"].append(zooniverse_ids)

                # what was the accuracy for this line?
                accuracy = len([c for c in merged_line if ord(c) != 27])/float(len(merged_line))
                subject_json["accuracy"].append(accuracy)

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

                subject_json["coordinates"].append([x_start,x_end,y_start,y_end])

        return subject_json





class AnnotateOutput(TranscriptionOutput):
    def __init__(self,project):
        TranscriptionOutput.__init__(self,project)

        # set up the reverse tags - so we can take a token (representing a full tag) and replace it with the
        # original tag
        self.tags = self.project.text_algorithm.tags
        self.reverse_tags = dict()

        for a,b in self.tags.items():
            self.reverse_tags[b] = a

    def __write_out_cluster__(self,cluster):
        """
        set up the json output for a single individual cluster
        :param cluster:
        :return:
        """
        if isinstance(cluster,list):
            print("problem cluster")
            cluster = cluster[0]

        cluster_to_json = {}
        # we need to retokenize everything

        tokenized_strings = []
        try:
            for annotation in cluster["cluster members"]:
                # extract the individual transcription
                i_transcription = annotation[-2]
                # convert from unicode to ascii
                i_transcription = i_transcription.encode('ascii','ignore')
                # replace each tag with a single character representation
                for chr_representation,tag in self.tags.items():
                    i_transcription = i_transcription.replace(tag,chr(chr_representation))

                tokenized_strings.append(i_transcription)
        except TypeError:
            print(cluster)
            print(type(cluster))
            raise

        # and now repeat with aggregated line
        aggregated_line = str(cluster["center"][-1])
        for chr_representation,tag in self.tags.items():
            aggregated_line = aggregated_line.replace(tag,chr(chr_representation))
        assert isinstance(aggregated_line,str)

        # store this cleaned aggregate text
        cluster_to_json["aggregated_text"] = self.__write_out_aggregate_line__(aggregated_line,tokenized_strings)
        # plus the coordinates
        cluster_to_json["coordinates"] = cluster["center"][:-1]
        # now add in the individual pieces of text
        cluster_to_json["individual transcriptions"] = self.__write_out_individual_lines__(cluster)

        return cluster_to_json

    def __write_out_individual_lines__(self,cluster):
        """
        set up the output for each individual transcription in a cluster
        :param cluster:
        :return:
        """
        individual_text_to_json = []
        for ii, annotation in enumerate(cluster["cluster members"]):
            # I think annotation[5] is no variants of words - for folger
            coords = annotation[:4]
            individual_text = annotation[4]
            assert isinstance(individual_text,unicode)

            # again, convert the tags to the ones needed by Folger or Tate (as opposed to the ones
            # zooniverse is using)
            # assert isinstance(individual_text,str)
            # for old,new in replacement_tags.items():
            #     individual_text = individual_text.replace(old,new)

            temp_text = individual_text
            skip = 0
            is_skip = False

            # we need to "rebuild" the individual text so that we can insert <skip>X</skip>
            # to denote that MAFFT inserted X spaces into the line
            individual_text = ""
            for c in temp_text:
                if ord(c) in [24,27]:
                    is_skip = True
                    skip += 1
                else:
                    if is_skip:
                        individual_text += "<skip>"+str(skip)+"</skip>"
                        skip = 0
                        is_skip = 0
                    individual_text += c

            individual_text_to_json.append({"coordinates":coords,"text":individual_text})

        return individual_text_to_json

    def __write_out_aggregate_line__(self,aggregated_line,tokenized_strings):
        """
        take care of setting up the output for one aggregate line
        :param tokenized_strings: so if there is disagreement, we can list the different options
        :return:
        """
        # convert some special characters and tokenize the disagreements

        line = ""

        agreement = True
        differences = {}

        for c_i,c in enumerate(aggregated_line):
            # if we have a disagreement (represented by either ord(c) = 27), keep on a running tally of
            # what all the differences are over all users so we can report the disagreements per "word"
            # of disagreement, not per character
            if ord(c) == 27:
                agreement = False

                # get all of the different transcriptions given by each user
                try:
                    char_options = [(ii,individual_text[c_i]) for ii,individual_text in enumerate(tokenized_strings)]
                except IndexError:
                    # print([len(t) for t in tokenized_strings])
                    # print(len(aggregated_line))
                    # print([c for c in aggregated_line])
                    # print("==---")
                    # for t in tokenized_strings:
                    #     print(t)
                    raise

                # add these characters to the running total - ord(c) == 24 is when MAFFT inserted
                # a space to align the text, which corresponds to not having anything there
                for ii,c in char_options:
                    if ord(c) != 24:
                        if ii not in differences:
                            differences[ii] = c
                        else:
                            differences[ii] += c
                    else:
                        if ii not in differences:
                            differences[ii] = ""
            elif ord(c) != 24:
                # if we just had a disagreement, print it out
                if not agreement:
                    line += "<disagreement>"
                    for options in set(differences.values()):
                        for tag,chr_representation in self.reverse_tags.items():
                            options = options.replace(chr(chr_representation),tag)
                        line += "<option>"+options+"</option>"
                    line += "</disagreement>"
                    differences = {}

                agreement = True

                # untokenize any tokens we find - replace them with the original tag
                for tag,chr_representation in self.reverse_tags.items():
                    c = c.replace(chr(chr_representation),tag)
                line += c

        # did we end on a disagreement?
        if not agreement:
            line += "<disagreement>"
            for c in set(differences.values()):
                for tag,chr_representation in self.reverse_tags.items():
                    c = c.replace(chr(chr_representation),tag)
                line += "<option>"+c+"</option>"
            line += "</disagreement>"

        # for the illegible tag, we added in some extra stuff, so now remove it
        line = line.replace(".*?","")
        return line

    def __subject_to_json__(self,subject_id,aggregation):
        """

        :param aggregation:
        :return subject_json:
        """
        subject_json = dict()
        # get the metadata for this subject
        subject_json["metadata"] = self.metadata[int(subject_id)]

        # are there any images in this subject?
        # there will always be "all_users" so we can looking for a list longer than one
        if ("image clusters" in aggregation["T2"]) and len(aggregation["T2"]["image clusters"]) > 1:
            subject_json["images"] = []
            for image_index,image in aggregation["T2"]["image clusters"].items():
                if image_index == "all_users":
                    continue
                subject_json["images"].append(image["center"])

        # are there any text clusters?
        if len(aggregation["T2"]["text clusters"]) >= 1:
            subject_json["text"] = []

            # now build up each one of the results
            for cluster_index,cluster in aggregation["T2"]["text clusters"].items():
                # this isn't really a cluster - more metadata, so skip it
                if cluster_index == "all_users":
                    continue

                # add this cluster to the total list
                subject_json["text"].append(self.__write_out_cluster__(cluster))

            # sort so they should appear in reading order
            subject_json["text"].sort(key = lambda x:x["coordinates"][2])

            # finally, give all of the individual transcriptions (removing alignment tags) without regards to
            # cluster - this way, people can tell if any text was ignored
            subject_json["raw transcriptions"] = self.__add_global_transcriptions__(subject_id)

        return subject_json


    def __add_global_transcriptions__(self,subject_id):
        """
        add the individual transcriptions without regards to the cluster - so people can tell if any
        text was ignored
        :return:
        """
        global_list = []

        workflow_id = self.project.workflows.keys()[0]

        for annotations in self.project.__sort_annotations__(workflow_id,[subject_id]):
            # annotation[1] is the list of markings for this subject - annotation[0] is the list
            # of classifications, [2] survey annotations and [3] image dimensions
            transcriptions = annotations[1]

            # make sure that this user actually provided transcriptions, otherwise, just skip
            if "T2" in transcriptions:
                for ii,(user_id,transcription,tool) in enumerate(transcriptions["T2"]["text"][subject_id]):
                    if transcription is None:
                        continue
                    coords = transcription[:4]
                    individual_text = transcription[4]
                    assert isinstance(individual_text,unicode)
                    if "\n" in individual_text:
                        continue

                    individual_text = individual_text.encode('ascii','ignore')
                    global_list.append({"coordinates":coords,"text":individual_text})

        return global_list