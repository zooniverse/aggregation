from __future__ import print_function
import numpy as np
import math

class Survey:
    def __init__(self):
        pass

    def __get_species_in_subject(self,aggregations):
        """
        use Ali's and Margaret's code to determine how many species are a given subject
        and return those X top species
        :return:
        """
        print(aggregations)
        num_species = int(np.median(aggregations["num species"]))
        assert(num_species >= 1)
        # sort the species by the number of votes
        species_by_vote = []

        for species_id in aggregations:
            if species_id not in ["num users","num species"]:
                species_by_vote.append((species_id,aggregations[species_id]["num votes"]))
        sorted_species = sorted(species_by_vote,key = lambda x:x[1],reverse=True)

        return sorted_species[:num_species]

    def __shannon_entropy__(self,probabilities):
        return -sum([p*math.log(p) for p in probabilities])

    def __calc__pielou__(self,aggregations):
        """
        return a row for survey tasks with subject id and pielou index
        :param aggregations:
        :return:
        """
        # for development - a few bad aggregations have crept into the db (for small projects) so skip them
        if max(aggregations["num species"]) == 0:
            return ""

        num_users = aggregations["num users"]
        probabilities = []
        for details in aggregations.values():
            # list => "num species", int => "num users"
            if type(details) in [list,int]:
                continue

            probabilities.append(details["num votes"]/float(num_users))

        num_species = len(probabilities)

        if num_species == 1:
            pielou_index = 0
        else:
            shannon = self.__shannon_entropy__(probabilities)
            pielou_index = shannon/math.log(num_species)

        return pielou_index

    def __aggregate__(self,raw_survey_classifications):
        # there should only be one task associated with Wildcam Gorongosa
        assert len(raw_survey_classifications.keys()) == 1
        task_id = raw_survey_classifications.keys()[0]

        aggregated_results = {}

        for subject_id in raw_survey_classifications[task_id]:
            annotation_dictionary = raw_survey_classifications[task_id][subject_id]

            # is this the first time we've dealt with this particular subject?
            # annotation_dictionary will have one key per user
            if subject_id not in aggregated_results:
                subject_results = {"num users":len(annotation_dictionary),"num species":[]}
            else:
                subject_results = aggregated_results[subject_id]

            # the user ids per annotation set don't actually matter - so just go through the dictionary values
            # i.e. the species seen
            for annotations_per_user in annotation_dictionary.values():
                # how many species did this one user see?
                subject_results["num species"].append(len(annotations_per_user))

                # go through each individual species the user marked/noted
                for species_annotation in annotations_per_user:
                    species = species_annotation["choice"]

                    # is the first classification we've encountered for this species?
                    if species not in subject_results:
                        subject_results[species] = {"num votes": 1,"followup":{}}

                    else:
                        subject_results[species]["num votes"] += 1

                    # go through each follow up question
                    for question,answer in species_annotation["answers"].items():
                        if question not in subject_results[species]["followup"]:
                            subject_results[species]["followup"][question] = {}

                        # if list - multiple answers are allowed
                        if isinstance(answer,list):
                            for ans in answer:
                                if ans not in subject_results[species]["followup"]:
                                    subject_results[species]["followup"][question][ans] = 1
                                else:
                                    subject_results[species]["followup"][question][ans] += 1
                        else:
                            if answer not in subject_results[species]["followup"]:
                                subject_results[species]["followup"][question][answer] = 1
                            else:
                                subject_results[species]["followup"][question][answer] += 1

                    if species not in ["NTHNGHR","FR"]:
                        assert "HWMN" in subject_results[species]["followup"]

            subject_results["num species in image"] = self.__get_species_in_subject(subject_results)
            subject_results["pielou index"] = self.__calc__pielou__(subject_results)

            aggregated_results[subject_id] = subject_results

        return aggregated_results
