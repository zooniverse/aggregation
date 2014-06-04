#!/usr/bin/env python
from __future__ import print_function
import csv
import pymongo


class IBCC:
    def __init__(self):
        self.client = pymongo.MongoClient()
        self.db = self.client['serengeti_2014-05-13']

        self.species_groups = [["gazelleThomsons","gazelleGrants"],]
        self.cutoff = 5

    def __csv_in__(self):
        reader = csv.reader(open("/home/ggdhines/Databases/serengeti/goldFiltered.csv","rb"), delimiter=" ")
        next(reader, None)

        curr_name = None
        curr_id = None
        species_list = []

        collection = self.db['merged_classifications']

        zooniverse_id_count = {}

        for row in reader:
                user_name = row[1]
                subject_zooniverse_id = row[2]
                species = row[11]

                if (user_name != curr_name) and (subject_zooniverse_id != curr_id):
                    if not(curr_name is None):
                        if curr_id in zooniverse_id_count:
                            zooniverse_id_count[curr_id] += 1
                        else:
                            zooniverse_id_count[curr_id] = 1

                        if zooniverse_id_count[curr_id] > self.cutoff:
                            continue

                        document = {"user_name": curr_name, "subject_zooniverse_id": curr_id, "species_list": species_list}

                        collection.insert(document)

                    curr_name = user_name
                    curr_species = []
                    curr_id = subject_zooniverse_id

                species_list.append(species)

        document = {"user_name": curr_name, "subject_zooniverse_id": curr_id, "species_list": species_list}
        collection.insert(document)

    def __IBCCoutput__(self,speciesGroup):
        collection = self.db['merged_classifications']
        requiredSpecies = ["gazelleThomsons"]
        prohibitedSpecies = ["gazelleGrants"]

        for document in collection.find({"species_list": {"$in": requiredSpecies}}):
            print(document)
            break

f = IBCC()
f.__csv_in__()
