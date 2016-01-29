__author__ = 'greg'
#every line must start off as - subject_id \t (GOLD if gold standard),
param_list = {"subject_id": 3, "time_stamp":4, "user_name":5, "x":14, "y":15, "animal":7, "label":11}

#what animals can we ignore when determining whether or not the photo is blank?
blank_types = ["carcassOrScale","carcass","other","blank"]

production = False

classifications_for_blank = 4
classifications_for_blank_consensus = 10