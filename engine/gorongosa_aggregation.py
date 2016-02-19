from __future__ import print_function

class Survey:
    def __init__(self):
        pass

    def __aggregate__(self,raw_classifications):
        # there should only be one task associated with Wildcam Gorongosa
        assert len(raw_classifications.keys()) == 1
        task_id = raw_classifications.keys()[0]

        aggregated_results = {}

        for subject_id in raw_classifications[task_id]:

            users_and_annotations = raw_classifications[task_id][subject_id]
            users,annotations = zip(*users_and_annotations)

            # is this the first time we've dealt with this particular subject?
            if subject_id not in aggregated_results:
                subject_results = {"num users":len(users),"num species":[]}
            else:
                subject_results = aggregated_results[subject_id]

            # how many species did this person see in this subject?
            num_species = len(annotations)
            subject_results["num species"].append(num_species)

            for ann in annotations:
                species = ann["choice"]

                # is the first classification we've encountered for this species?
                if species not in subject_results:
                    subject_results[species] = {"num votes": 1,"followup":{}}

                else:
                    subject_results[species]["num votes"] += 1

                for question,answer in ann["answers"].items():
                    if question not in subject_results[species]["followup"]:
                        subject_results[species]["followup"][question] = {}

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

            aggregated_results[subject_id] = subject_results

        return aggregated_results
