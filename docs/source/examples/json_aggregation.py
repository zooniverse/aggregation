from __future__ import print_function
import json

with open('/home/ggdhines/Downloads/tmp/376.json') as data_file:
    aggregation_results = json.load(data_file)

print(aggregation_results.keys())