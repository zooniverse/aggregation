__author__ = 'greg'
import bisect
import math
import abc
import numpy as np
import os
import re
# import ibcc
import csv
import matplotlib.pyplot as plt
import random
import socket
import warnings
import scipy
from shapely.geometry import Polygon
# import panoptes_api


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


# code from
# http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(dict(seq[int(last):int(last + avg)]))
        last += avg

    return out




class Cluster:
    __metaclass__ = abc.ABCMeta

    def __init__(self,shape):
        """
        :param project_api: how to talk to whatever project we are clustering for (Panoptes/Ouroboros shouldn't matter)
        :param min_cluster_size: minimum number of points in a cluster to not be considered noise
        :return:
        """
        self.shape = shape

        # assert isinstance(project_api,panoptes_api.PanoptesAPI)
        # self.project_api = project_api
        # we must say what shapes we want
        # self.clusterResults = {}
        # self.goldResults = {}

        # self.correct_pts = {}
        # for gold points which we have missed
        # self.missed_pts = {}
        # for false positives (according to the gold standard)
        # self.false_positives = {}
        # we also need to map between user points and gold points
        # self.user_gold_mapping = {}

        # the distance between a user cluster and what we believe to do the corresponding goal standard point
        # self.user_gold_distance = {}

        # the list of every subject we have processed so far - a set just in case we accidentally process the same
        # one twice. Probably should never happen but just in case
        # self.processed_subjects = set([])

        # needed for when we want use ibcc for stuff
        # current_directory = os.getcwd()
        # slash_indices = [m.start() for m in re.finditer('/', current_directory)]
        # self.base_directory = current_directory[:slash_indices[2]+1]

        # for creating names of ibcc output files
        # self.alg = None

        self.x_roc = None
        self.y_roc = None

        self.algorithm_name = None

    @abc.abstractmethod
    def __inner_fit__(self,markings,user_ids,tools):
        """
        the main function for clustering
        :param user_ids:
        :param markings:
        :param jpeg_file:
        :return cluster_centers: the center of each cluster - probably just take the average along each dimension
        feel free to try something else but the results might not mean as much
        :return markings_per_cluster: the markings in each cluster
        :return users_per_cluster: the user id of each marking per cluster
        :return time_to_cluster: how long it took to cluster
        """
        cluster_centers = []
        markings_per_cluster = []
        users_per_cluster = []
        time_to_cluster = 0

        return (cluster_centers , markings_per_cluster, users_per_cluster), time_to_cluster

    def __aggregate__(self,raw_markings):
        """
        the function to call from outside to do the clustering
        override but call if you want to add additional functionality
        :param subject_id: what is the subject (in Ouroboros == zooniverse_id)
        :param jpeg_file: for debugging - to show step by step what is happening
        :return:
        """
        aggregation = {"param":"subject_id"}
        # start by calling the api to get the annotations along with the list of who made each marking
        # so for this function, we know that annotations = markings
        # all_markings =  self.project_api.__get_markings__(subject_id,gold_standard)
        # print all_markings
        # self.clusterResults[subject_id] = {"param":"task_id"}
        for task_id in raw_markings:
            # go through each shape independently
            for shape in raw_markings[task_id].keys():
                # if is this shape does not correspond to the specific shape this clustering algorithm was
                # created for - skip
                if shape != self.shape:
                    continue

                for subject_id in raw_markings[task_id][shape]:
                    assert raw_markings[task_id][shape][subject_id] != []

                    # remove any "markings" which correspond to the user not making a marking
                    # these are still useful for noting that the user saw that image
                    pruned_markings = [(u,m,t) for u,m,t in raw_markings[task_id][shape][subject_id] if m is not None]
                    all_users,t1,t2 = zip(*raw_markings[task_id][shape][subject_id])
                    all_users = list(set(all_users))

                    # empty image
                    if pruned_markings == []:
                        # no centers, no points, no users per cluster
                        cluster_results = []
                    else:
                        users,markings,tools = zip(*pruned_markings)

                        # do the actual clustering
                        cluster_results,time_to_cluster = self.__inner_fit__(markings,users,tools)

                    # store the results - note we need to store even for empty images
                    if subject_id not in aggregation:
                        aggregation[subject_id] = {"param":"task_id"}
                    if task_id not in aggregation[subject_id]:
                        aggregation[subject_id][task_id] = {"param":"clusters"}
                    if shape not in aggregation[subject_id][task_id]:
                        # store the set of all users who have seen this subject/task
                        # used for determining false vs. true positives
                        aggregation[subject_id][task_id][str(shape) + " clusters"] = {"param":"cluster_index","all_users":all_users}

                    for cluster_index,cluster in enumerate(cluster_results):
                        aggregation[subject_id][task_id][shape+ " clusters"][cluster_index] = cluster

        # we should have some results
        # assert aggregation != {"param":"subject_id"}
        return aggregation

