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
            aggregated_results[subject_id] = {}

            users_and_annotations = raw_classifications[task_id][subject_id]
            users,annotations = zip(*users_and_annotations)

            aggregated_results[subject_id]["num_users"] = len(users)

            num_species_per_user = [len(ann) for ann in annotations]
            num_species = int(np.median(num_species_per_user))

            subject_results = {}
            for ann in annotations:
                for species_ann in ann:
                    species = species_ann["choice"]

                    # is the first classification we've encountered for this species?
                    if species not in aggregated_results[subject_id]:
                        subject_results[species] = {"num votes": 1,"followup answers":{}}

                    else:
                        subject_results[species] += 1

                    for question,answer in species_ann["answers"].items():
                        if question not in aggregated_results[subject_id][species]["followup"]:
                            subject_results[species]["followup"][question] = {}

                        if isinstance(answer,list):
                            for ans in answer:
                                if ans not in subject_results[species]["followup"]:
                                    subject_results[species]["followup"][ans] = 1
                                else:
                                    subject_results[species]["followup"][ans] += 1
                        else:
                            if answer not in subject_results[species]["followup"]:
                                subject_results[species]["followup"][answer] = 1
                            else:
                                subject_results[species]["followup"][answer] += 1

            aggregated_results[subject_id] = subject_results
            print aggregated_results[subject_id]

            # report all species - we'll prune later
            # sorted_species = sorted(species_vote.items(),key = lambda x:x[1],reverse=True)
            # top_votes = sorted_species[:num_species]
            # if top_votes == []:
            #     continue
            # top_species,count = zip(*top_votes)

            # for species in species_vote:
            #     # record how many people voted for each species
            #     followup_answers = {"num votes":species_vote[species]}
            #     for ann in annotations:
            #         for species_ann in ann:
            #             if species_ann["choice"] == species:
            #
            #                 for question,answer in species_ann["answers"].items():
            #
            #                     if question not in followup_answers:
            #                         followup_answers[question] = {}
            #
            #                     if isinstance(answer,list):
            #                         for ans in answer:
            #                             if ans not in followup_answers[question]:
            #                                 followup_answers[question][ans] = 1
            #                             else:
            #                                 followup_answers[question][ans] += 1
            #                     else:
            #                         if answer not in followup_answers[question]:
            #                             followup_answers[question][answer] = 1
            #                         else:
            #                             followup_answers[question][answer] += 1
            #
            #     aggregated_results[subject_id][species] = followup_answers

        return aggregated_results
