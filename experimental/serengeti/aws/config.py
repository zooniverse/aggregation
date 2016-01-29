__author__ = 'greg'
project_type = "animalMarking"
#every line must start off as - subject_id \t (GOLD if gold standard),
param_list = {"subject_id": 2, "time_stamp":4, "user_name":1, "animal":11}

#what animals can we ignore when determining whether or not the photo is blank?
blank_types = [""]

production = False

classifications_for_blank = 4
classifications_for_blank_consensus = 10