import numpy as np


class WildcamGorongosaSurvey:
    def __init__(self):
        pass

    def __aggregate__(self,raw_classifications):
        assert len(raw_classifications.keys()) == 1

        task_id = raw_classifications.keys()[0]

        aggregated_results = {}

        for subject_id in raw_classifications[task_id]:
            aggregated_results[subject_id] = {}

            users_and_annotations = raw_classifications[task_id][subject_id]
            users,annotations = zip(*users_and_annotations)

            aggregated_results[subject_id]["num_users"] = len(users)

            num_species_per_user = [len(ann) for ann in annotations]
            num_species = int(np.median(num_species_per_user))

            species_vote = {}

            # annotations is the list of all annotations made by all users who have this subject
            for ann in annotations:
                species = ann["choice"]

                if species not in species_vote:
                    species_vote[species] = 1
                else:
                    species_vote[species] += 1

            # what are the most likely species?
            sorted_species = sorted(species_vote.items(),key = lambda x:x[1],reverse=True)
            top_votes = sorted_species[:num_species]
            if top_votes == []:
                continue
            top_species,count = zip(*top_votes)

            for species in top_species:
                followup_answers = {}
                for ann in annotations:
                    if ann["choice"] == species:

                        for question,answer in ann["answers"].items():

                            if question not in followup_answers:
                                followup_answers[question] = {}

                            if isinstance(answer,list):
                                for ans in answer:
                                    if ans not in followup_answers[question]:
                                        followup_answers[question][ans] = 1
                                    else:
                                        followup_answers[question][ans] += 1
                            else:
                                if answer not in followup_answers[question]:
                                    followup_answers[question][answer] = 1
                                else:
                                    followup_answers[question][answer] += 1

                aggregated_results[subject_id][species] = followup_answers

        return aggregated_results

class Survey:
    def __init__(self):
        pass

    def __aggregate__(self,raw_classifications):
        assert len(raw_classifications.keys()) == 1

        task_id = raw_classifications.keys()[0]

        aggregated_results = {}

        for subject_id in raw_classifications[task_id]:
            aggregated_results[subject_id] = {}

            users_and_annotations = raw_classifications[task_id][subject_id]
            users,annotations = zip(*users_and_annotations)

            aggregated_results[subject_id]["num_users"] = len(users)

            num_species_per_user = [len(ann) for ann in annotations]
            num_species = int(np.median(num_species_per_user))

            species_vote = {}
            for ann in annotations:
                print annotations
                for species_ann in ann:
                    species = species_ann["choice"]

                    if species not in species_vote:
                        species_vote[species] = 1
                    else:
                        species_vote[species] += 1

            # what are the most likely species?
            sorted_species = sorted(species_vote.items(),key = lambda x:x[1],reverse=True)
            top_votes = sorted_species[:num_species]
            if top_votes == []:
                continue
            top_species,count = zip(*top_votes)

            for species in top_species:
                followup_answers = {}
                for ann in annotations:
                    for species_ann in ann:
                        if species_ann["choice"] == species:

                            for question,answer in species_ann["answers"].items():

                                if question not in followup_answers:
                                    followup_answers[question] = {}

                                if isinstance(answer,list):
                                    for ans in answer:
                                        if ans not in followup_answers[question]:
                                            followup_answers[question][ans] = 1
                                        else:
                                            followup_answers[question][ans] += 1
                                else:
                                    if answer not in followup_answers[question]:
                                        followup_answers[question][answer] = 1
                                    else:
                                        followup_answers[question][answer] += 1

                aggregated_results[subject_id][species] = followup_answers

        return aggregated_results
