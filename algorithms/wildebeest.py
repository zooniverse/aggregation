#!/usr/bin/env python
import sys
sys.path.append("/home/greg/github/reduction/engine")
sys.path.append("/home/ggdhines/PycharmProjects/reduction/engine")

__author__ = 'greg'
from aggregation_api import AggregationAPI
import numpy

workflow_id = 6


wildebeest = AggregationAPI(6)
aggregations = wildebeest.__aggregate__(workflows = [6],store_values=False)

marking_task = wildebeest.workflows[workflow_id][1].keys()[0]
tools = wildebeest.workflows[workflow_id][1][marking_task]

workflows,versions,instructions,updated_at_timestamps = wildebeest.__get_workflow_details__(workflow_id)
tools_labels = instructions[workflow_id][marking_task]["tools"]

for j,subject_id in enumerate(aggregations):
    overall_votes = {int(t_index): [] for t_index in range(len(tools))}
    for annotation in wildebeest.__get_raw_classifications__(subject_id,workflow_id):
        tool_votes = {int(t_index): 0 for t_index in range(len(tools))}
        for task in annotation:
            if task["task"] == marking_task:
                for marking in task["value"]:
                    tool_votes[int(marking["tool"])] += 1
        for t_index in tool_votes:
            overall_votes[t_index].append(tool_votes[t_index])
    print
    print "===----"
    print subject_id
    print len(aggregations[subject_id][marking_task]["point clusters"]) - 2
    for cluster_index in aggregations[subject_id][marking_task]["point clusters"]:
        if cluster_index in ["param","all_users"]:
            continue

        cluster = aggregations[subject_id][marking_task]["point clusters"][cluster_index]
        print cluster["tool_classification"],cluster["existence"]

    print

    for t_index,v in overall_votes.items():
        # print str(tools_labels[t_index]["marking tool"]) + "\t\t\t" + str(numpy.mean(v))
        print str(t_index) + "\t\t\t" + str(numpy.mean(v)) + "\t\t\t" + str(numpy.median(v))


    if j== 10:
        break