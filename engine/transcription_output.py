import json
from text_aggregation import TranscriptionAPI
from helper_functions import warning
import tarfile
import numpy as np
"""
sets up the output for transcription projects like Annotate and Shakespeare's world
before being emailed out
"""
__author__ = 'ggdhines'


class TranscriptionOutput:
    def __init__(self,project):
        assert isinstance(project,TranscriptionAPI)
        self.project = project

    def __json_output__(self):
        pass

    def __tar_output__(self):
        aws_tar = self.project.__get_aws_tar_name__()
        print("saving json results")
        with tarfile.open("/tmp/"+aws_tar,mode="w") as t:
            t.add("/tmp/"+str(self.project.project_id)+".json")



class ShakespearesWorldOutput(TranscriptionOutput):
    def __init__(self,project):
        TranscriptionOutput.__init__(self,project)

    def __json_output__(self):
        print("restructing json results")
        workflow_id = self.project.workflows.keys()[0]

        cur = self.project.postgres_session.cursor()

        stmt = "select subject_id,aggregation from aggregations where workflow_id = " + str(workflow_id)
        cur.execute(stmt)

        new_json = {}

        subjects_with_results = 0

        for ii,(subject_id,aggregation) in enumerate(cur.fetchall()):
            #
            if subject_id not in self.project.classification_alg.to_retire:
                continue
            try:
                clusters_by_line = {}

                if isinstance(aggregation,str):
                    print("converting aggregation to string")
                    aggregation = json.loads(aggregation)

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
                        if subject_id not in new_json:
                            new_json[subject_id] = {"text":[],"individual transcriptions":[], "accuracy":[], "coordinates" : [],"users_per_line":[]}

                            # add in the metadata
                            metadata = self.project.__get_subject_metadata__(subject_id)["subjects"][0]["metadata"]
                            new_json[subject_id]["metadata"] = metadata

                            new_json[subject_id]["zooniverse subject id"] = subject_id

                        # add in the line of text
                        new_json[subject_id]["text"].append(merged_line)

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
                        new_json[subject_id]["individual transcriptions"].append(rep_cluster["aligned_text"])
                        new_json[subject_id]["users_per_line"].append(zooniverse_ids)

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

        json.dump(new_json,open("/tmp/"+str(self.project.project_id)+".json","wb"))
        self.__tar_output__()



class AnnotateOutput(TranscriptionOutput):
    def __init__(self,project):
        TranscriptionOutput.__init__(self,project)

    def __json_output__(self):
        aggregations_to_json = dict()
        # get the list of all retired subjects
        # retired_subjects = project.__get_subjects__(workflow_id,only_retired_subjects=True)

        workflow_id = self.project.workflows.keys()[0]

        for subject_id,aggregations in list(self.project.__yield_aggregations__(workflow_id))[:10]:
            print subject_id

            # an empty subject is represented by not having an image value or text value
            aggregations_to_json[subject_id] = {}

            # get the metadata for this subject
            m = self.project.__get_subject_metadata__(subject_id)
            metadata = m["subjects"][0]["metadata"]
            # metadata = json.dumps(metadata)
            aggregations_to_json[subject_id]["metadata"] = metadata


            # are there any images in this subject?
            # there will always be "all_users" so we can looking for a list longer than one
            if ("image clusters" in aggregations["T2"]) and len(aggregations["T2"]["image clusters"]) > 1:
                aggregations_to_json[subject_id]["images"] = []
                for image_index,image in aggregations["T2"]["image clusters"].items():
                    if image_index == "all_users":
                        continue
                    aggregations_to_json[subject_id]["images"].append(image["center"])

            # are there any text clusters?
            if len(aggregations["T2"]["text clusters"]) > 1:
                aggregations_to_json[subject_id]["text"] = []

                # now build up each one of the results
                for cluster_index,cluster in aggregations["T2"]["text clusters"].items():

                    if cluster_index == "all_users":
                        continue
                    if "Ward" in cluster["center"][-1]:
                        print [individual_text for (coord,individual_text) in enumerate(cluster["cluster members"])]
                        for (coord,text) in cluster["cluster members"]:
                            print text
                            print [(ii,individual_text[c_i]) for ii,(coord,individual_text) in enumerate(cluster["cluster members"])]
                        print
                    # print cluster
                    # assert False

                    cluster_to_json = {}

                    # for folger this will allow us to remove sw- from all of the tags
                    # for both folger and annotate, we will set <unclear>.*</unclear> to just <unclear></unclear>
                    # aggregated_line = str(cluster["center"][-1])

                    # we need to retokenize everything
                    # print aggregated_line
                    tags = self.project.text_algorithm.tags
                    reverse_tags = dict()

                    for a,b in tags.items():
                        reverse_tags[b] = a

                    tokenized_strings = []
                    for _,l_m in cluster["cluster members"]:
                        l_m = l_m.encode('ascii','ignore')
                        for tag,chr_representation in tags.items():
                            l_m = l_m.replace(tag,chr(chr_representation))
                            # aggregated_line = aggregated_line.replace(old,chr(new))

                        tokenized_strings.append(l_m)

                    aggregated_line = str(cluster["center"][-1])
                    # print type(aggregated_line)
                    assert isinstance(aggregated_line,str)

                    # convert some special characters and tokenize the disagreements
                    line = ""

                    agreement = True
                    differences = {}

                    # we need to treat tags as a single character
                    # convert the aggregated line to tokenized tags so that it and all
                    # of the individual lines are the same length
                    try:
                        for chr_representation,tag in reverse_tags.items():
                            aggregated_line = aggregated_line.replace(chr(chr_representation),tag)
                    except UnicodeDecodeError:
                        print aggregated_line
                        raise

                    for c_i,c in enumerate(aggregated_line):
                        # if we have a disagreement (represented by either ord(c) = 27), keep on a running tally of
                        # what all the differences are over all users so we can report the disagreements per "word"
                        # of disagreement, not per character
                        if ord(c) == 27:
                            agreement = False

                            # get all of the different transcriptions given by each user
                            try:

                                # print [(ii,individual_text[c_i]) for ii,(coord,individual_text) in enumerate(cluster["cluster members"])]
                                # print "***"
                                char_options = [(ii,individual_text[c_i]) for ii,individual_text in enumerate(tokenized_strings)]
                            except IndexError:
                                print aggregated_line
                                print len(aggregated_line)
                                print [len(l[1]) for l in cluster["cluster members"]]
                                for l in cluster["cluster members"]:
                                    print l
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
                                    for chr_representation,tag in reverse_tags.items():
                                        options = options.replace(chr(chr_representation),tag)
                                    line += "<option>"+options+"</option>"
                                line += "</disagreement>"
                                differences = {}

                            agreement = True

                            for chr_representation,tag in reverse_tags.items():
                                c = c.replace(chr(chr_representation),tag)
                            line += c

                    # did we end on a disagreement?
                    if not agreement:
                        line += "<disagreement>"
                        for c in set(differences.values()):
                            for chr_representation,tag in reverse_tags.items():
                                c = c.replace(chr(chr_representation),tag)
                            line += "<option>"+c+"</option>"
                        line += "</disagreement>"

                    # if "Ward" in line:
                    #     print line
                    #     print cluster["cluster members"]
                    #     print json.dumps(cluster["cluster members"],sort_keys=True,indent=4, separators=(',', ': '))
                    #     assert False

                    # store this cleaned aggregate text
                    cluster_to_json["aggregated_text"] = line
                    # plus the coordinates
                    cluster_to_json["coordinates"] = cluster["center"][:-1]

                    # now add in the individual pieces of text
                    individual_text_to_json = []
                    for ii,(coords,individual_text) in enumerate(cluster["cluster members"]):
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

                    cluster_to_json["individual transcriptions"] = individual_text_to_json
                    aggregations_to_json[subject_id]["text"].append(cluster_to_json)

                # sort so they should appear in reading order
                aggregations_to_json[subject_id]["text"].sort(key = lambda x:x["coordinates"][2])


                # finally, give all of the individual transcriptions (removing alignment tags) without regards to
                # cluster - this way, people can tell if any text was ignored
                transcriptions = self.project.__sort_annotations__(workflow_id,[subject_id])[1]
                aggregations_to_json[subject_id]["raw transcriptions"] = []
                for ii,(user_id,transcription,tool) in enumerate(transcriptions["T2"]["text"][subject_id]):
                    if transcription is None:
                        continue
                    coords = transcription[:-1]
                    individual_text = transcription[-1]
                    if "\n" in individual_text:
                        continue

                    individual_text = individual_text.encode('ascii','ignore')
                    aggregations_to_json[subject_id]["raw transcriptions"].append({"coordinates":coords,"text":individual_text})

        project_id = str(self.project.project_id)
        with open("/tmp/"+project_id+".json", 'w') as outfile:
            # print aggregations_to_json
            # print json.dumps(aggregations_to_json, sort_keys=True,indent=4, separators=(',', ': '))
            # assert False
            json.dump(aggregations_to_json,outfile)

        self.__tar_output__()