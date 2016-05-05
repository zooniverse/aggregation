from __future__ import print_function
import json
import numpy as np
with open('/home/ggdhines/Downloads/tmp/376.json') as data_file:
    aggregation_results = json.load(data_file)

print(aggregation_results["1273551"])

# accuracy_list = []
# for subject in  aggregation_results.values():
#     if subject["accuracy"] != []:
#         accuracy_list.append(np.mean(subject["accuracy"]))
#
# print(np.mean(accuracy_list))
# # print(aggregation_results.keys())
