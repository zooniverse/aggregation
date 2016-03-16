from __future__ import print_function
import numpy as np

class Survey:
    def __init__(self):
        pass

    def __aggregate__(self,raw_classifications):

        # todo - figure out why this is happening
        if raw_classifications == {}:
            return dict()

        assert len(raw_classifications.keys()) == 1

        task_id = raw_classifications.keys()[0]

        aggregated_results = {}

        for subject_id in raw_classifications[task_id]:
            # just to save on some typing
            subject_results = {"num species":[]}
            subject_classifications = raw_classifications[task_id][subject_id]

            num_users = 0

            for user_id,annotations in subject_classifications.items():
                # how many species did this user report?
                num_species = len([a for a in annotations if a != []])
                # if the user just reported nothing - skip
                if num_species == 0:
                    continue

                num_users += 1

                subject_results["num species"].append(num_species)

                for ann in annotations:
                    # not sure hy we have sometimes getting empty annotations but it happens - if so, skip
                    if ann == []:
                        continue

                    for species_ann in ann:
                        species = species_ann["choice"]

                        # is the first classification we've encountered for this species?
                        if species not in subject_results:
                            subject_results[species] = {"num votes": 1,"followup":{}}

                        else:
                            subject_results[species]["num votes"] += 1

                        # go through each of the follow up questions
                        for question,answer in species_ann["answers"].items():
                            # is this the first time we've seen this particular followup question
                            # should probably always happen the first time we see the species (since most, if not
                            # all of the time, all of the follow up questions should be required)
                            if question not in subject_results[species]["followup"]:
                                subject_results[species]["followup"][question] = {}

                            # if list - multiple answers are allowed
                            if isinstance(answer,list):
                                for ans in answer:
                                    # first time we've seen this particular answer?
                                    if ans not in subject_results[species]["followup"]:
                                        subject_results[species]["followup"][question][ans] = 1
                                    else:
                                        subject_results[species]["followup"][question][ans] += 1
                            else:
                                # first time we've seen this particular answer
                                if answer not in subject_results[species]["followup"]:
                                    subject_results[species]["followup"][question][answer] = 1
                                else:
                                    subject_results[species]["followup"][question][answer] += 1

            # save the results and record how many people have seen this particular subject
            # on the off chance that all of the annotations were empty, skip it
            if subject_results["num species"] != []:
                aggregated_results[subject_id] = subject_results
                aggregated_results[subject_id]["num users"] = num_users

        return aggregated_results
