#!/usr/bin/env python
__author__ = 'greg'
from sklearn.cluster import DBSCAN
import numpy as np
import math
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook
from copy import deepcopy
import warnings
import time
import random
from clustering import Cluster
import abc

class TooBig(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return None

class NodeTemplate:
    def __init__(self, key, markings, users,r_users=[],r_markings=[]):
        self.key = key
        self.users = users
        self.users.extend(r_users)
        self.markings = markings
        self.markings.extend(r_markings)

        self.size = len(self.markings)

    def __get_cluster__(self):
        return self.markings,self.users

class AgglomerativeNode(NodeTemplate):
    def __init__(self,left_node,right_node):
        NodeTemplate.__init__(self,left_node.key,left_node.markings,left_node.users,right_node.users,right_node.markings)
        assert left_node.key < right_node.key
        #self.left_node = left_node
        #self.right_node = right_node

class AgglomerativeLeaf(NodeTemplate):
    def __init__(self,pt,user):
        NodeTemplate.__init__(self,pt,[pt],[user])
        self.x,self.y = pt

        # rare but it happens - some users can give the EXACT same x and y values for a marking
        # this causes a problem with using the pts as keys
        # so let's add a very small random offset - no guarantee that it will work but pretty sure
        x,y = pt
        x += random.uniform(-0.0001,0.0001)
        y += random.uniform(-0.0001,0.0001)
        self.key = x,y

class Agglomerative(Cluster):
    def __init__(self, project_api,min_cluster_size=1):
        Cluster.__init__(self,project_api,min_cluster_size)
        self.distances = {}
        self.dbscan_preprocess = False

    def __set_distances__(self,clusters):
        for c1 in clusters:
            self.distances[c1.key] = {}
            for c2 in clusters:
                if c1 == c2:
                    continue

                # check to see if there is any overlap
                if c1.users != c2.users:
                    # the distances should be symmetric so we only need to keep one way
                    if c1.key < c2.key:
                        dist = math.sqrt((c1.x-c2.x)**2+(c1.y-c2.y)**2)
                        self.distances[c1.key][c2.key] = dist

    def dist(self, c1,c2):
        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    class CannotSplit(Exception):
        def __init__(self,value):
            self.value = value
        def __str__(self):
            return ""

    def flatten(self, l):
        if isinstance(l,list):
            if len(l) == 0:
                return []
            first, rest = l[0],l[1:]
            return self.flatten(first) + self.flatten(rest)
        else:
            return [l]


    def __agglomerate_clusters__(self,clusters,fname = None):
        """
        agglomerative clusters - keep going until none of the clusters can be merged
        use fname if we want to see step by step pictures
        :todo: - make sure that fname is properly supported
        :param clusters:
        :param fname:
        :return:
        """

        # keep going while there is more than one cluster - we can stop earlier than that
        while len(clusters) > 1:
            # we want the merger with minimum distance
            objective_func = float("inf")
            bestChoice = None

            if not(fname is None):
                print [c.users for c in clusters]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image_file = cbook.get_sample_data(fname)
                    image = plt.imread(image_file)

                    fig, ax = plt.subplots()
                    im = ax.imshow(image)
                    for c in clusters:
                        X,Y = zip(*c.markings)
                        cent_x = np.mean(X)
                        cent_y = np.mean(Y)
                        plt.plot(cent_x,cent_y,".",color="blue")

                    plt.show()

            # look at merging every possible pair of clusters
            for i,c1 in enumerate(clusters):
                for j,c2 in enumerate(clusters[i+1:]):
                    try:
                        clusterDist = self.__get_dist__(c1.key,c2.key)
                    except KeyError:
                        # not stored - so there should be an overlap between the users
                        continue

                    # have we improved on the objective function
                    if clusterDist < objective_func:
                            objective_func = clusterDist
                            bestChoice = (i,i+1+j)
            if not(fname is None):
                print objective_func

            # if there is no merging to be done - stop
            if bestChoice is None:
                break

            # heuristics to avoid select clusters from opposite sides of the images
            # todo: make it so that you can turn the heuristics on and off
            # if (objective_func/float(len(c1.markings)+len(c2.markings))) > (5*objective_func):
            #     break
            # old_objective = objective_func/float(len(c1.markings)+len(c2.markings))


            # the second one comes later, so pop it first so we don't mess up  indices
            # note that this does not mean that the key for c1 is lower than c2
            c2 = clusters.pop(bestChoice[1])
            c1 = clusters.pop(bestChoice[0])

            # update the distances and create a new merged node
            if c1.key < c2.key:
                self.__update_distances__(clusters,c1,c2)
                clusters.append(AgglomerativeNode(c1,c2))
            else:
                self.__update_distances__(clusters,c2,c1)
                clusters.append(AgglomerativeNode(c2,c1))

            # a bit of error checking - commented out but might still be useful
            # basically checking to make sure that we are only keeping distances for pairs of nodes with no overlap
            # for c in clusters:
            #     if c1.key < c2.key:
            #         if c.key == c1.key:
            #             continue
            #
            #         try:
            #             self.__get_dist__(c.key,c1.key)
            #         except KeyError:
            #             overlap = [u for u in c.users if u in c1.users]
            #             if overlap == []:
            #                 raise
            #     else:
            #         if c.key == c2.key:
            #             continue
            #
            #         try:
            #             self.__get_dist__(c.key,c2.key)
            #         except KeyError:
            #             overlap = [u for u in c.users if u in c2.users]
            #             if overlap == []:
            #                 raise

        return clusters

    def __get_dist__(self,key1,key2):
        """
        return a distance
        :param key1:
        :param key2:
        :return:
        """
        if key1 < key2:
            return self.distances[key1][key2]
        else:
            return self.distances[key2][key1]

    def __delete_dist__(self,key1,key2):
        """
        delete a distance if we no longer need it - so one of the keys' cluster is going away
        :param key1:
        :param key2:
        :return:
        """
        if key1 < key2:
            del self.distances[key1][key2]
        else:
            del self.distances[key2][key1]

    def __set_distance__(self,key1,key2,dist):
        """
        set the distance between two clusters identified by their keys
        since the distance from a to b is equal to the distance from b to a, we only need to store one of them
        so let's only store based on the lower key value
        :param key1:
        :param key2:
        :param dist:
        :return:
        """
        if key1 < key2:
            self.distances[key1][key2] = dist
        else:
            self.distances[key2][key1] = dist

    def __update_distances__(self,clusters,c_i,c_j):
        """
        update the distances between clusters as the result of merging c_i and c_j. c_j is going to become part
        of c_i. Use  Lance Williams algorithm. For documentation see
        http://en.wikipedia.org/wiki/Ward%27s_method
        or watch
        https://www.youtube.com/watch?v=aXsaFNVzzfI
        :param clusters:
        :param c_i:
        :param c_j: c_j is the cluster that we going to  delete (or at least it will  become part of c_i)
        :return:
        """
        # we need c_i to have the smaller key so switch if necessary. This is the result of using points in clusters
        # as a key for that cluster
        if c_i.key > c_j.key:
            temp = c_j
            c_j = c_i
            c_i = temp

        assert c_i.key < c_j.key

        for c_k in clusters:
            # the distance from c_i or c_j to c_i \cup c_j is going to be zero so worry about that later
            if (c_k.key == c_i.key) or (c_k.key == c_j.key):
                pass

            # is there an overlap between clusters?
            # if there is already an overlap between c_k and c_i, then there shouldn't be any values stored
            # but if there is an overlap between c_k and c_j, we need to just delete it
            overlap_ki = [u for u in c_k.users if u in c_i.users]
            overlap_kj = [u for u in c_k.users if u in c_j.users]


            # if we have both overlaps - there shouldn't be anything we need to do
            if (overlap_ki != []) and (overlap_kj != []):
                # print "1"
                pass
            elif overlap_kj != []:
                # overlap with c_j, which means that distance[c_k][c_j] shouldn't exist in the first place
                # and as a result of the merge, we shouldn't have distance[c_k][c_i] either
                self.__delete_dist__(c_k.key,c_i.key)

            elif overlap_ki != []:
                # overlap with c_i so distance[c_k][c_i] shouldn't exist. Get rid of distance[c_k][c_j] since it will
                # be going away anyways
                self.__delete_dist__(c_k.key,c_j.key)
            else:
                # no overlaps
                # since we are only storing the distances values when c < c', we need to be slightly cute about this
                # the following are the actual terms of the Lance Williams equation
                t = self.alpha(c_i.size, c_j.size, c_k.size)*self.__get_dist__(c_i.key,c_k.key)
                t += self.alpha(c_j.size, c_i.size, c_k.size)*self.__get_dist__(c_j.key,c_k.key)
                t += self.beta(c_i.size, c_j.size, c_k.size)*self.distances[c_i.key][c_j.key]
                t += self.gamma()* math.fabs(self.__get_dist__(c_i.key,c_k.key) - self.__get_dist__(c_j.key,c_k.key))

                # update the distances
                self.__set_distance__(c_i.key,c_k.key,t)

                # c_j is going to disappear so delete any distance values associated with it
                # note we didn't delete until now since we needed this value for the update
                self.__delete_dist__(c_k.key,c_j.key)

        # finish removing any references to c_j
        del self.distances[c_j.key]

    @abc.abstractmethod
    def alpha(self,i,j,k):
        """
        calculate the alpha value between clusters i,j and k - j and k are the ones that are about to be merged
        :param i:
        :param j:
        :param k:
        :return:
        """
        return 0

    @abc.abstractmethod
    def beta(self,i,j,k):
        """
        calculate the beta value between clusters i,j and k - j and k are the ones that are about to be merged
        :param i:
        :param j:
        :param k:
        :return:
        """
        return 0

    @abc.abstractmethod
    def gamma(self):
        """
        calculate the gamma value
        :return:
        """
        return 0

    def __fit__(self,markings,user_ids,jpeg_file=None):
        # associate each marking with their corresponding user and extract only the relevant part of the marking for
        # the clustering
        l = [[(u,m[0]) for m in marking] for u,marking in zip(user_ids,markings)]
        l_flattened = [item for sublist in l for item in sublist]

        end_clusters =[]
        start = time.time()
        if self.dbscan_preprocess:
            # todo: add dbscan preprocessing
            pass

        # start by creating a singleton cluster for each cluster
        starting_clusters = [AgglomerativeLeaf(pts,user) for user,pts in l_flattened]
        # then init the distances between each cluster
        self.__set_distances__(starting_clusters)
        # then agglomerative the clusters
        end_clusters = self.__agglomerate_clusters__(starting_clusters)

        # convert each tree into a list of pts and users - the cluster which this tree represents
        results = [node.__get_cluster__() for node in end_clusters]
        markings,users = zip(*results)

        # find the center of each cluster
        centers = []
        for cluster in markings:
            X,Y = zip(*cluster)
            centers.append((np.mean(X),np.mean(Y)))
        end = time.time()

        return (centers, markings,users), end-start


class Ward(Agglomerative):
    """
    choose alpha, beta and gamma values which result in using the Ward distance to merge clusters
    """
    def __init__(self, project_api,min_cluster_size=1):
        Agglomerative.__init__(self,project_api,min_cluster_size)

    def alpha(self,i,j,k):
        return (i+k)/float(i+j+k)

    def beta(self,i,j,k):
        return -k/float(i+j+k)

    def gamma(self):
        return 0