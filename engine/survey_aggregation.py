from __future__ import print_function
import math
import numpy as np

class Survey:
    def __init__(self):
        pass

    def __get_species_in_subject__(self,aggregations):
        """
        use Ali's and Margaret's code to determine how many species are a given subject
        and return those X top species
        :return:
        """
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

    def __species_annotation__(self,aggregation_so_far,annotation):
        """
        for a given user's annotation - go through each species that the user marked and process that annotation
        :param aggregation_so_far: aggregations so far for a single subject
        :param annotation: the next annotation for that subject
        :return:
        """
        for species_ann in annotation:
            species = species_ann["choice"]

            # is the first classification we've encountered for this species?
            if species not in aggregation_so_far:
                aggregation_so_far[species] = {"num votes": 1,"followup":{}}

            else:
                aggregation_so_far[species]["num votes"] += 1

            # go through each of the follow up questions
            for question,answer in species_ann["answers"].items():
                # is this the first time we've seen this particular followup question
                # should probably always happen the first time we see the species (since most, if not
                # all of the time, all of the follow up questions should be required)
                if question not in aggregation_so_far[species]["followup"]:
                    aggregation_so_far[species]["followup"][question] = {}

                # if list - multiple answers are allowed
                if isinstance(answer,list):
                    for ans in answer:
                        # first time we've seen this particular answer?
                        if ans not in aggregation_so_far[species]["followup"]:
                            aggregation_so_far[species]["followup"][question][ans] = 1
                        else:
                            aggregation_so_far[species]["followup"][question][ans] += 1
                else:
                    # first time we've seen this particular answer
                    if answer not in aggregation_so_far[species]["followup"]:
                        aggregation_so_far[species]["followup"][question][answer] = 1
                    else:
                        aggregation_so_far[species]["followup"][question][answer] += 1

        return aggregation_so_far


    def __aggregate__(self,raw_classifications,aggregated_results={}):

        # todo - figure out why this is happening
        if raw_classifications == {}:
            return dict()

        assert len(raw_classifications.keys()) == 1

        task_id = raw_classifications.keys()[0]

        for subject_id in raw_classifications[task_id]:
            # just to save on some typing
            annotation_dictionary = raw_classifications[task_id][subject_id]
            # some annotations appear to be empty - those should just be skipped
            # as a result, don't calculate "num users" just yet
            subject_results = {"num species":[]}

            num_users = 0

            for annotations_per_user in annotation_dictionary.values():
                # how many species did this user report?
                num_species = len([a for a in annotations_per_user if a != []])
                # if the user just reported nothing - skip
                if num_species == 0:
                    continue

                num_users += 1

                subject_results["num species"].append(num_species)
                # assert False

                # go through each individual species annotation from user
                for ann in annotations_per_user:
                    # not sure why we have sometimes getting empty annotations but it happens - if so, skip
                    if ann == []:
                        continue

                    subject_results = self.__species_annotation__(subject_results,ann)

            # save the results and record how many people have seen this particular subject
            # on the off chance that all of the annotations were empty, skip it
            if subject_results["num species"] != []:
                subject_results["num users"] = num_users
                subject_results["num species in image"] = self.__get_species_in_subject__(subject_results)
                subject_results["pielou index"] = self.__calc__pielou__(subject_results)


                aggregated_results[subject_id] = subject_results

        return aggregated_results
