#!/usr/bin/env python

__author__ = 'greg'
from aggregation_api import AggregationAPI
import classification
import matplotlib.pyplot as plt
import cluster_count

workflow_id = 6


class Wildebeest(AggregationAPI):
    def __init__(self):
        AggregationAPI.__init__(self,"wildebeest")
        self.cluster_algs["point"] = cluster_count.Counting("point")

    # def __plot__(self,workflow_id,task):
    #     print self.description[task]
    #     for subject_id in self.subject_sets[workflow_id]:
    #         if subject_id in self.classification_alg.results:
    #
    #             # self.__plot_individual_points__(subject_id,task)
    #             # self.__plot_cluster_results__(subject_id,task)
    #             # plt.title("number of users: " + str(len(all_users)))
    #             classifications = self.classification_alg.results[subject_id][task]
    #             # print classifications
    #             votes,total = classifications
    #             if total >= 5:
    #                 self.__plot_image__(subject_id)
    #                 title = ""
    #                 for answer_index,percentage in votes.items():
    #                     if title != "":
    #                         title += "\n"
    #                     title += self.description[task][answer_index+1] + ": " + str(int(percentage*total))
    #                 plt.title(title)
    #
    #                 plt.savefig("/home/greg/Databases/wildebeest/where_is_the_sun/"+str(subject_id)+".jpg")
    #                 plt.close()

ali = Wildebeest()
print ali.__setup_workflows__()
# ali.__aggregate__(workflows=[workflow_id])