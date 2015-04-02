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

        self.correct_pts = {}
        # for gold points which we have missed
        self.missed_pts = {}
        # for false positives (according to the gold standard)
        self.false_positives = {}

    def __display__markings__(self, subject_id):
        """
        display the results of clustering algorithm - the image must already have been downloaded
        :param subject_id:
        :param fname: the file name of the downloaded image
        :return:
        """
        x,y = zip(*self.clusterResults[subject_id][0])
        args = [x,y,'o']
        kwargs = {"color":"red"}

        ax = self.project_api.__display_image__(subject_id,[args],[kwargs])

    def __display_results__(self,subject_id):
        # green is for correct points
        x,y = zip(*self.correct_pts[subject_id])
        args_l = [[x,y,'o'],]
        kwargs_l = [{"color":"green"},]

        # yellow is for missed points
        x,y = zip(*self.missed_pts[subject_id])
        args_l.append([x,y,'o'])
        kwargs_l.append({"color":"yellow"})

        # red is for false positives
        print self.false_positives[subject_id]
        x,y = zip(*self.false_positives[subject_id])
        args_l.append([x,y,'o'])
        kwargs_l.append({"color":"red"})

        ax = self.project_api.__display_image__(subject_id,args_l,kwargs_l)

    @abc.abstractmethod
    def __fit__(self,markings,user_ids,jpeg_file=None):
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

    def __cluster_subject__(self,subject_id,jpeg_file=None):
        """
        the function to call from outside to do the clustering
        override but call if you want to add additional functionality
        :param subject_id: what is the subject (in Ouroboros == zooniverse_id)
        :param jpeg_file: for debugging - to show step by step what is happening
        :return:
        """
        # start by calling the api to get the annotations along with the list of who made each marking
        # so for this function, we know that annotations = markings
        users, markings = self.project_api.__get_markings__(subject_id)

        # if there are any markings - cluster
        # otherwise, just set to empty
        if markings != []:
            cluster_results,time_to_cluster = self.__fit__(markings,users,jpeg_file)
        else:
            cluster_results = [],[],[]
            time_to_cluster = 0

        self.clusterResults[subject_id] = cluster_results

        return time_to_cluster

    def __find_correct_markings__(self,subject_id):
        """
        find which user clusters we think are correct points
        :param subject_id:
        :return:
        """
        # get the markings made by the experts
        gold_markings = self.project_api.__get_markings__(subject_id,expert_markings=True)
        # extract the actual points
        gold_pts = zip(*gold_markings[1][0])[0]

        self.correct_pts[subject_id] = []
        self.missed_pts[subject_id] = []

        cluster_centers = self.clusterResults[subject_id][0]

        # if either of these sets are empty, then by def'n we can't have any correct WRT this image
        if (cluster_centers == []) or (gold_pts == []):
            return

        # user to gold - for a gold point X, what are the user points for which X is the closest gold point?
        userToGold = [[] for i in range(len(gold_pts))]
        # and the same from gold to user
        goldToUser = [[] for i in range(len(cluster_centers))]

        # find which gold standard pts, the user cluster pts are closest to
        # this will tell us which gold points we have actually found
        for local_index, u_pt in enumerate(cluster_centers):
            # dist = [math.sqrt((float(pt["x"])-x)**2+(float(pt["y"])-y)**2) for g_pt in gold_pts]
            min_dist = float("inf")
            closest_gold_index = None

            # find the nearest gold point to the cluster center
            # doing this in a couple of lines so that things are simpler - need to allow
            # for an arbitrary number of dimensions
            for gold_index,g_pt in enumerate(gold_pts):
                dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip(u_pt,g_pt)]))

                if dist < min_dist:
                    min_dist = dist
                    closest_gold_index = gold_index

            userToGold[closest_gold_index].append(local_index)

        # and now find out which user clusters are actually correct
        # that will be the user point which is closest to the gold point
        for gold_index,g_pt in enumerate(gold_pts):
            min_dist = float("inf")
            closest_user_index = None

            for u_index in userToGold[gold_index]:
                dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip(cluster_centers[u_index],g_pt)]))

                if dist < min_dist:
                    min_dist = dist
                    closest_user_index = u_index

            # if none then we haven't found this point
            if closest_user_index is not None:
                self.correct_pts[subject_id].append(cluster_centers[closest_user_index])
            else:
                self.missed_pts[subject_id].append(g_pt)

        # what were the false positives?
        self.false_positives[subject_id] = [pt for pt in cluster_centers if not(pt in self.correct_pts[subject_id])]

        print self.correct_pts[subject_id]
        print self.false_positives[subject_id]
        # for local_index, u_pt in enumerate(cluster_centers):
        #     # dist = [math.sqrt((float(pt["x"])-x)**2+(float(pt["y"])-y)**2) for g_pt in gold_pts]
        #     min_dist = float("inf")
        #     closest_gold_pt = None
        #
        #     # find the nearest gold point to the cluster center
        #     # doing this in a couple of lines so that things are simpler - need to allow
        #     # for an arbitrary number of dimensions
        #     for gold_index,g_pt in enumerate(gold_pts):
        #         dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip (u_pt,g_pt)]))
        #
        #         if dist < min_dist:
        #             min_dist = dist
        #             closest_gold_pt = gold_index
        #
        #     userToGold[closest_gold_pt].append(local_index)
        #
        # for gold_index, pt in enumerate(gold_pts):
        #     dist = [math.sqrt((float(pt["x"])-x)**2+(float(pt["y"])-y)**2) for (x,y) in cluster_centers]
        #     goldToUser[dist.index(min(dist))].append(gold_index)
        #
        # for local_index in range(len(cluster_centers)):
        #     if len(goldToUser[local_index]) >= 1:
        #             self.correct_pts[subject_id].append(local_index)



