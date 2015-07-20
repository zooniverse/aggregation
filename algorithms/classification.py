__author__ = 'greg'
import clustering
import ouroboros_api
import os
import re
import matplotlib.pyplot as plt
import networkx as nx
import itertools

import abc
import json
import csv

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

class Classification:
    def __init__(self,clustering_alg=None):
        # assert isinstance(project,ouroboros_api.OuroborosAPI)

        if clustering_alg is not None:
            assert isinstance(clustering_alg,clustering.Cluster)
        self.cluster_alg = clustering_alg

        current_directory = os.getcwd()
        slash_indices = [m.start() for m in re.finditer('/', current_directory)]
        self.base_directory = current_directory[:slash_indices[2]+1]
        # print self.base_directory

        self.species = {"lobate":0,"larvaceanhouse":0,"salp":0,"thalasso":0,"doliolidwithouttail":0,"rocketthimble":1,"rockettriangle":1,"siphocorncob":1,"siphotwocups":1,"doliolidwithtail":1,"cydippid":2,"solmaris":2,"medusafourtentacles":2,"medusamorethanfourtentacles":2,"medusagoblet":2,"beroida":3,"cestida":3,"radiolariancolonies":3,"larvacean":3,"arrowworm":3,"shrimp":4,"polychaeteworm":4,"copepod":4}
        self.candidates = self.species.keys()

    @abc.abstractmethod
    def __task_aggregation__(self,classifications,gold_standard=False):
        return []

    def __subtask_classification__(self):
        # todo: implement this
        assert False
        # we are dealing with tasks
        # is shape uncertain - if so - only accept markings from some users - who used the "correct" tool
        if "shapes" in classification_tasks[task_id]:
            assert False
        else:
            for shape in classification_tasks[task_id]["shapes"]:
                # create a temporary set of classifications
                shape_classification = {}

                for subject_id in raw_classifications[task_id][shape]:
                    # print raw_classifications[task_id][shape][subject_id]
                    # print subject_id
                    # print raw_classifications[task_id][shape].keys()
                    # print clustering_results[task_id][shape].keys()
                    # assert subject_id in clustering_results[task_id][shape]
                    # look at the individual points in the cluster
                    for cluster_index in clustering_results[subject_id][task_id][shape]:
                        if cluster_index == "param":
                            continue

    def __existence_classification__(self,task_id,raw_classifications,clustering_results):
        """
        classify whether clusters are true or false positives
        i.e. whether each cluster corresponds to something which actually exists

        return in json format so we can merge with other results
        :return:
        """
        aggregations = {}

        # raw_classifications and clustering_results have different hierarchy orderings- raw_classifications
        # is better for processing data and clustering_results is better for showing the end result
        # technically we only need to look at the data from clustering_results right now but its
        # hierarchy is really inefficient so use raw_classifications to help

        # each shape is done independently
        for shape in raw_classifications[task_id]:
            if shape == "param":
                continue
            # pretentious name but basically whether each person who has seen a subject thinks it is a true
            # positive or not
            existence_classification = {"param":"subject_id"}

            global_cluster_index = 0
            clusters_per_subject = []

            # look at the individual points in the cluster
            for subject_id in raw_classifications[task_id][shape]:
                if subject_id == "param":
                    continue
                if subject_id not in clustering_results:
                    continue
                clusters_per_subject.append([])

                # in either case probably an empty image
                if subject_id not in clustering_results:
                    continue
                if task_id not in clustering_results[subject_id]:
                    continue

                for local_cluster_index in clustering_results[subject_id][task_id][shape]:
                    if (local_cluster_index == "param") or (local_cluster_index == "all_users"):
                        continue

                    # extract the users who marked this cluster
                    cluster = clustering_results[subject_id][task_id][shape][local_cluster_index]
                    users = cluster["users"]

                    ballots = []

                    for u in clustering_results[subject_id][task_id][shape]["all_users"]:
                        if u in users:
                            ballots.append((u,1))
                        else:
                            ballots.append((u,0))

                    existence_classification[(subject_id,local_cluster_index)] = ballots
                    clusters_per_subject[-1].append(global_cluster_index)
                    global_cluster_index += 1

            existence_results = self.__task_aggregation__(existence_classification)
            assert isinstance(existence_results,dict)

            for subject_id,cluster_index in existence_results:
                if subject_id not in aggregations:
                    aggregations[subject_id] = {"param":"task_id"}
                if task_id not in aggregations[subject_id]:
                    aggregations[subject_id][task_id] = {"param":"shape"}
                if shape not in aggregations[subject_id][task_id]:
                    aggregations[subject_id][task_id][shape] = {}
                # this part is probably redundant
                if cluster_index not in aggregations[subject_id][task_id][shape]:
                    aggregations[subject_id][task_id][shape][cluster_index] = {}

                aggregations[subject_id][task_id][shape][cluster_index]["existence"] = existence_results[(subject_id,cluster_index)]

        return aggregations

    def __tool_classification__(self,task_id,classification_tasks,raw_classifications,clustering_results):
        """
        if multiple tools can make the same shape - we need to decide which tool actually corresponds to this cluster
        for example if both the adult penguin and chick penguin make a pt - then for a given point we need to decide
        if it corresponds to an adult or chick
        :param task_id:
        :param classification_tasks:
        :param raw_classifications:
        :param clustering_results:
        :return:
        """

        if clustering_results == {"param":"subject_id"}:
            print "warning - empty classifications"
            return {}

        aggregations = {}
        # are there any uncertain shapes associated with this task?
        # if not, return an empty result
        if "shapes" not in classification_tasks[task_id]:
            return {}

        # only go through the "uncertain" shapes
        for shape in classification_tasks[task_id]["shapes"]:
            tool_classifications = {}
            for subject_id in raw_classifications[task_id][shape]:
                # look at the individual points in the cluster

                # this should only happen if there were badly formed markings
                if raw_classifications[task_id][shape][subject_id] == {}:
                    continue

                # in either case probably an empty image
                if subject_id not in clustering_results:
                    continue
                if task_id not in clustering_results[subject_id]:
                    continue

                for cluster_index in clustering_results[subject_id][task_id][shape]:
                    if (cluster_index == "param") or (cluster_index == "all_users"):
                        continue

                    cluster = clustering_results[subject_id][task_id][shape][cluster_index]
                    pts = cluster["points"]
                    users = cluster["users"]
                    # users = clustering_results[subject_id][task_id][shape][subject_id]["users"]

                    # in this case, we want to "vote" on the tools
                    ballots = []
                    for (p,user) in zip(pts,users):
                        try:
                            tool_index = raw_classifications[task_id][shape][subject_id][(tuple(p),user)]
                        except KeyError:
                            print "===----"
                            print cluster
                            print raw_classifications[task_id][shape][subject_id].keys()
                            print (tuple(p),user)
                            raise

                        ballots.append((user,tool_index))

                    tool_classifications[(subject_id,cluster_index)] = ballots

            # classify
            tool_results = self.__task_aggregation__(tool_classifications)
            assert isinstance(tool_results,dict)

            for subject_id,cluster_index in tool_results:
                if subject_id not in aggregations:
                    aggregations[subject_id] = {"param":"task_id"}
                if task_id not in aggregations[subject_id]:
                    aggregations[subject_id][task_id] = {"param":"shape"}
                if shape not in aggregations[subject_id][task_id]:
                    aggregations[subject_id][task_id][shape] = {}
                # this part is probably redundant
                if cluster_index not in aggregations[subject_id][task_id][shape]:
                    aggregations[subject_id][task_id][shape][cluster_index] = {}

                aggregations[subject_id][task_id][shape][cluster_index]["shape_classification"] = tool_results[(subject_id,cluster_index)]

        return aggregations

    def __aggregate__(self,raw_classifications,workflow,clustering_results=None):
        # use the first subject_id to find out which tasks we are aggregating the classifications for
        aggregations = {"param":"subject_id"}
        classification_tasks,marking_tasks = workflow

        for task_id in classification_tasks:
            # print task_id
            if isinstance(classification_tasks[task_id],bool):
                # we have a basic classification task
                print task_id,classification_tasks[task_id]
                print
                print raw_classifications[task_id]
                temp_results = self.__task_aggregation__(raw_classifications[task_id])
                # convert everything into a dict and add the the task id
                task_results = {}
                for subject_id in temp_results:
                    task_results[subject_id] = {task_id:temp_results[subject_id]}
            else:
                # we have classifications associated with markings
                # make sure we have clustering results associated with these classifications
                assert clustering_results is not None

                # we have to first decide which cluster is a "true positive" and which is a "false positive"
                # so a question of whether or not people marked it - regardless of whether they marked it as the
                # correct "type"
                existence_results = self.__existence_classification__(task_id,raw_classifications,clustering_results)

                # now decide what type each cluster is
                # note that this step does not care whether a cluster is a false positive or not (i.e. the results
                # from the previous step are not taken into account)
                tool_results = self.__tool_classification__(task_id,classification_tasks,raw_classifications,clustering_results)

                task_results = self.__merge_results__(existence_results,tool_results)

                # are there any subtasks associated with this task/marking?
                if "subtask" in classification_tasks[task_id]:
                    self.__subtask_classification__()

            assert isinstance(task_results,dict)
            for subject_id in task_results:
                if subject_id not in aggregations:
                    aggregations[subject_id] = {"param":"task_id"}
                # we have results from other tasks, so we need to merge in the results
                assert isinstance(task_results[subject_id],dict)
                aggregations[subject_id] = self.__merge_results__(aggregations[subject_id],task_results[subject_id])
                # aggregations[subject_id][task_id] = task_results[subject_id]



        return aggregations

    def __merge_results__(self,r1,r2):
        """
        if we have json results from two different steps of the aggregation process - merge them
        :param r1:
        :param r2:
        :return:
        """
        assert isinstance(r1,dict)
        assert isinstance(r2,dict)

        for kw in r2:
            try:
                if kw not in r1:
                    r1[kw] = r2[kw]
                elif r1[kw] != r2[kw]:
                    r1[kw] = self.__merge_results__(r1[kw],r2[kw])
            except TypeError:
                print "==--"
                print r1
                print r2
                print kw
                raise
        return r1


class VoteCount(Classification):
    def __init__(self,clustering_alg=None):
        Classification.__init__(self,clustering_alg)

    def __task_aggregation__(self,raw_classifications,gold_standard=False):
        """
        question_id is not None if and only if the classification relates to a marking
        :param subject_ids:
        :param task_id:
        :param question_id:
        :param gold_standard:
        :return:
        """
        results = {}

        for subject_id in raw_classifications:
            vote_counts = {}
            if subject_id == "param":
                continue
            # print raw_classifications[subject_id]
            for user,ballot in raw_classifications[subject_id]:
                if ballot is None:
                    continue
                # in which case only one vote is allowed
                if isinstance(ballot,int):
                    if ballot in vote_counts:
                        vote_counts[ballot] += 1
                    else:
                        vote_counts[ballot] = 1
                # in which case multiple votes are allowed
                else:
                    for vote in ballot:
                        if vote in vote_counts:
                            vote_counts[vote] += 1
                        else:
                            vote_counts[vote] = 1
            # convert to percentages
            percentages = {}
            for vote in vote_counts:
                percentages[vote] = vote_counts[vote]/float(sum(vote_counts.values()))

            results[subject_id] = percentages,sum(vote_counts.values())

        return results


