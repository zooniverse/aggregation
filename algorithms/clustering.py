__author__ = 'greg'
import pymongo
import bisect
import sys
import os
import csv
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook
from collections import Iterator
import math
from scipy.stats.stats import pearsonr
import cPickle as pickle
from scipy.stats.mstats import normaltest
import warnings
import time
import ouroboros_api
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import abc

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# # for Greg - which computer am I on?
# import ibcc
# import multiClickCorrect



def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError






class Cluster:
    __metaclass__ = abc.ABCMeta

    def __init__(self, project_api,min_cluster_size=1):
        """
        :param project_api: how to talk to whatever project we are clustering for (Panoptes/Ouroboros shouldn't matter)
        :param min_cluster_size: minimum number of points in a cluster to not be considered noise
        :return:
        """
        assert isinstance(project_api,ouroboros_api.MarkingProject)
        self.project_api = project_api
        self.min_cluster_size = min_cluster_size
        self.clusterResults = {}

    @abc.abstractmethod
    def __fit__(self,user_ids,markings,jpeg_file=None):
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

    def __cluster_subject__(self,subject_id,jpeg_file=None,more_to_do=False):
        """
        the function to call from outside to do the clustering
        override but call if you want to add additional functionality
        :param subject_id: what is the subject (in Ouroboros == zooniverse_id)
        :param jpeg_file: for debugging - to show step by step what is happening
        :param more_to_do: by default, cluster_subject will return the results directly to the project_api class
        but we might have more we want to do with the results before sending back
        :return:
        """
        # start by calling the api to get the annotations along with the list of who made each marking
        # so for this function, we know that annotations = markings
        users, markings = self.project_api.__get_cluster_annotations__()

        # if there are any markings - cluster
        # otherwise, just set to empty
        if markings != []:
            cluster_results,time_to_cluster = self.__fit__(markings,users,jpeg_file)

        else:
            cluster_results = [],[],[]
            time_to_cluster = 0

        return cluster_results,time_to_cluster

