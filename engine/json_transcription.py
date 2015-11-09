#!/usr/bin/env python
__author__ = 'ggdhines'
from transcription import Tate
import json
from latex_transcription import get_updated_tags
from copy import deepcopy

project_id = 245
environment = "development"
workflow_id = 121
transcription_task = "T2"

# project_id = 376
# workflow_id = 205

first = True
count = 0

replacement_tags = get_updated_tags(project_id)

aggregations_to_json = {}
from transcription import retired_subjects

with Tate(project_id,environment) as project:
    # get the list of all retired subjects
    # retired_subjects = project.__get_subjects__(workflow_id,only_retired_subjects=True)

    for subject_id in retired_subjects:
        # there will only be one aggregation per subject so a loop would be slightly silly
        # instead I'll just cast to a list and take the first element
        aggregation_list = list(project.__yield_aggregations__(workflow_id,subject_id=subject_id))
        print subject_id

        # an empty subject is represented by not having an image value or text value
        aggregations_to_json[subject_id] = {}

        # get the metadata for this subject
        m = project.__get_subject_metadata__(subject_id)
        metadata = m["subjects"][0]["metadata"]
        metadata = json.dumps(metadata)
        aggregations_to_json[subject_id]["metadata"] = metadata

        if len(aggregation_list) == 0:
            # these should be empty subjects
            # todo - double check
            print "empty subject :: " + str(subject_id)
            continue

        _,aggregation = aggregation_list[0]

        # are there any images in this subject?
        # there will always be "all_users" so we can looking for a list longer than one
        if len(aggregation["T2"]["image clusters"]) > 1:
            aggregations_to_json[subject_id]["images"] = []
            for image_index,image in aggregation["T2"]["image clusters"].items():
                if image_index == "all_users":
                    continue
                aggregations_to_json[subject_id]["images"].append(image["center"])

        # are there any text clusters?
        if len(aggregation["T2"]["text clusters"]) > 1:
            aggregations_to_json[subject_id]["text"] = []

            # now build up each one of the results
            for cluster_index,cluster in aggregation["T2"]["text clusters"].items():
                if cluster_index == "all_users":
                    continue

                cluster_to_json = {}

                # for folger this will allow us to remove sw- from all of the tags
                # for both folger and annotate, we will set <unclear>.*</unclear> to just <unclear></unclear>
                aggregated_line = cluster["center"][-1]

                for old,new in replacement_tags.items():
                    aggregated_line = aggregated_line.replace(old,new)

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
                            char_options = [(ii,individual_text[c_i]) for ii,(coord,individual_text) in enumerate(cluster["cluster members"])]
                        except IndexError:
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
                    else:
                        # if we just had a disagreement, print it out
                        if not agreement:
                            line += "<disagreement>"
                            for c in set(differences.values()):
                                line += "<option>"+c+"</option>"
                            line += "</disagreement>"
                            differences = {}

                        agreement = True
                        line += c
                # did we end on a disagreement?
                if not agreement:
                    line += "<disagreement>"
                    for c in set(differences.values()):
                        line += "<option>"+c+"</option>"
                    line += "</disagreement>"

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
                    for old,new in replacement_tags.items():
                        individual_text = individual_text.replace(old,new)

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

            # finally, give all of the individual transcriptions (removing alignment tags) without regards to
            # cluster - this way, people can tell if any text was ignored
            transcriptions = project.__sort_annotations__(workflow_id,[subject_id])[1]
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

with open('/home/ggdhines/tate.json', 'w') as outfile:
    json.dump(aggregations_to_json,outfile)
