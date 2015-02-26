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

class Agglomerative:
    def __init__(self, min_samples=1):
        self.min_samples = min_samples
        self.distances = {}

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

    # def __get_clusters__(self,forest):
    #     for tree in forest:
    #         tree.__get_cluster__()

    # def getClusters(self, hierarchy):
    #     centers = []
    #     clusters = []
    #     users = []
    #
    #     if isinstance(hierarchy,tuple):
    #         #print "here - grrrrr"
    #         centers = [(hierarchy[0],hierarchy[1])]
    #         clusters = [[(hierarchy[0],hierarchy[1])]]
    #         users = [[hierarchy[2], ], ]
    #     elif self.averageLinkage(hierarchy[0],hierarchy[1]) == float("inf"):
    #         centers,clusters,users = self.getClusters(hierarchy[0])
    #         #clusters.extend(self.getClusters(hierarchy[0]))
    #         #clusters.extend(self.getClusters(hierarchy[1]))
    #         centers_rhs,clusters_rhs,users_rhs = self.getClusters(hierarchy[1])
    #         centers.extend(centers_rhs)
    #         clusters.extend(clusters_rhs)
    #         users.extend(users_rhs)
    #     else:
    #         newCluster = self.flatten(hierarchy)
    #         if len(newCluster) >= 1:
    #             x,y,u = zip(*newCluster)
    #             # cluster_centers, end_clusters,end_users
    #             #clusters = [((np.mean(x),np.mean(y)),zip(x,y),u)]
    #             centers = [(np.mean(x),np.mean(y))]
    #             clusters = [zip(x,y)]
    #             users = [u]
    #         else:
    #             print "too small"
    #
    #     return centers,clusters,users

    def averageLinkage(self,c1,c2):
        d = []
        flattened1 = self.flatten(c1)
        flattened2 = self.flatten(c2)

        for pt1 in flattened1:
            x1,y1,u1 = pt1
            for pt2 in flattened2:
                x2,y2,u2 = pt2
                if u1 == u2:
                    d.append(float("inf"))
                else:
                    d.append(math.sqrt((x2-x1)**2+(y2-y1)**2))

        return np.mean(d)

    # def mergeClusters(self,c1,c2):
    #     retval = c1[:]
    #     retval.extend(c2)
    #
    #     return retval
    #
    # def findClusterCenter(self,cluster):
    #     x,y,u = zip(*self.flatten(cluster))
    #     return np.mean(x),np.mean(y)

    def __agglomerate_clusters__(self,clusters,fname = None):
        while len(clusters) > 1:

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

            for i,c1 in enumerate(clusters):
                for j,c2 in enumerate(clusters[i+1:]):
                    try:
                        if c1.key < c2.key:
                            clusterDist = self.distances[c1.key][c2.key]
                        else:
                            clusterDist = self.distances[c2.key][c1.key]
                    except KeyError:
                        # not stored - so there should be an overlap between the users
                        continue

                    if clusterDist < objective_func:
                            objective_func = clusterDist
                            bestChoice = (i,i+1+j)
            if not(fname is None):
                print objective_func
            # if there is no merging to be done
            if bestChoice is None:
                break

            # the second one comes later, so pop it first so we don't mess up  indices
            c2 = clusters.pop(bestChoice[1])
            c1 = clusters.pop(bestChoice[0])


            #print c1.users,c2.users

            if c1.key < c2.key:
                self.__update_distances__(clusters,c1,c2)
                clusters.append(AgglomerativeNode(c1,c2))
            else:
                self.__update_distances__(clusters,c2,c1)
                clusters.append(AgglomerativeNode(c2,c1))


            for c in clusters:
                if c1.key < c2.key:
                    if c.key == c1.key:
                        continue

                    try:
                        self.__get_dist__(c.key,c1.key)
                    except KeyError:
                        overlap = [u for u in c.users if u in c1.users]
                        if overlap == []:
                            raise
                else:
                    if c.key == c2.key:
                        continue

                    try:
                        self.__get_dist__(c.key,c2.key)
                    except KeyError:
                        overlap = [u for u in c.users if u in c2.users]
                        if overlap == []:
                            raise

        return clusters

    def __get_dist__(self,key1,key2):
        if key1 < key2:
            try:
                return self.distances[key1][key2]
            except KeyError:
                # print
                # print
                # print sorted(self.distances.keys(), key = lambda x:x[0])
                # print key1,key2
                # print self.distances[key1].keys()
                # print self.distances[key2].keys()
                raise
        else:
            try:
                return self.distances[key2][key1]
            except KeyError:
                # print
                # print
                # print sorted(self.distances.keys(), key = lambda x:x[0])
                # print key1, key2
                # print sorted(self.distances[key1].keys())
                # print sorted(self.distances[key2].keys())
                raise

    def __delete_dist__(self,key1,key2):
        if key1 < key2:
            try:
                del self.distances[key1][key2]
            except KeyError:
                # print self.distances.keys()
                # print self.distances[key1].keys()
                raise
        else:
            try:
                del self.distances[key2][key1]
            except KeyError:
                # print self.distances.keys()
                # print self.distances[key2].keys()
                raise

    def __set_distance__(self,key1,key2,dist):
        if key1 < key2:
            self.distances[key1][key2] = dist
        else:
            self.distances[key2][key1] = dist

    def __update_distances__(self,clusters,c_i,c_j):
        # c_j is the cluster that we going to  delete (or at least it will  become part of c_i)
        #print (c_i.key,c_j.key)
        # make sure we have the right order
        if c_i.key > c_j.key:
            temp = c_j
            c_j = c_i
            c_i = temp

        assert c_i.key < c_j.key
        for c_k in clusters:
            if (c_k.key == c_i.key) or (c_k.key == c_j.key):
                pass

            # is there an overlap?
            # if there is already an overlap between c_k and c_i, then there shouldn't be any values stored
            # but if there is an overlap between c_k and c_j, we need to just delete it
            overlap_ki = [u for u in c_k.users if u in c_i.users]
            overlap_kj = [u for u in c_k.users if u in c_j.users]


            # if we have both overlaps - there shouldn't be anything we need to do
            if (overlap_ki != []) and (overlap_kj != []):
                # print "1"
                pass
            elif overlap_kj != []:
                # print "2"
                # overlap with c_j, which means that distance[c_k][c_j] shouldn't exist in the first place
                # and as a result of the merge, we shouldn't have distance[c_k][c_i] either
                self.__delete_dist__(c_k.key,c_i.key)

            elif overlap_ki != []:
                # print "3"
                # overlap with c_i so distance[c_k][c_i] shouldn't exist. Get rid of distance[c_k][c_j] since it will
                # be going away anyways
                self.__delete_dist__(c_k.key,c_j.key)
            else:
                # print "4"
                # no overlaps

                # since we are only storing the distances values when c < c', we need to be slightly cute about this
                t = 0
                #t += self.alpha(c_i.size, c_j.size, c_k.size)(c_i.size+c_k.size)/total_size*self.__get_dist__(c_i.key,c_k.key)
                t += self.alpha(c_i.size, c_j.size, c_k.size)*self.__get_dist__(c_i.key,c_k.key)
                #t += (c_j.size+c_k.size)/total_size*self.__get_dist__(c_j.key,c_k.key)
                t += self.alpha(c_j.size, c_i.size, c_k.size)*self.__get_dist__(c_j.key,c_k.key)
                #t += self.beta(c_i.size + c_j.size + c_k.size)-c_k.size/total_size*self.distances[c_i.key][c_j.key]
                t += self.beta(c_i.size, c_j.size, c_k.size)*self.distances[c_i.key][c_j.key]
                try:
                    t += self.gamma()* math.fabs(self.__get_dist__(c_i.key,c_k.key) - self.__get_dist__(c_j.key,c_k.key))
                except KeyError:
                    print self.__get_dist__(c_i.key,c_k.key)
                    print self.__get_dist__(c_j.key,c_k.key)
                    raise

                # update the distances
                self.__set_distance__(c_i.key,c_k.key,t)

                # delete the value if we are storing it
                self.__delete_dist__(c_k.key,c_j.key)

        del self.distances[c_j.key]

    def alpha(self,i,j,k):
        assert False

    def beta(self,i,j,k):
        assert False

    def gamma(self):
        assert False

    def __total_distance__(self,c1,c2):
        dist = []
        for (x1,y1) in c1.markings:
            for (x2,y2) in c2.markings:
                dist.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

        return sum(dist)

    def __average_distance__(self,c1,c2):
        dist = []
        for (x1,y1) in c1.markings:
            for (x2,y2) in c2.markings:
                dist.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

        return np.mean(dist)

    # def __agglomerate_clusters2__(self,clusters):
    #     #
    #     while len(clusters) > 1:
    #             minAvg = float("inf")
    #             bestChoice = None
    #
    #             for i,c1 in enumerate(clusters):
    #                 for j,c2 in enumerate(clusters[i+1:]):
    #                     clusterDist = self.averageLinkage(c1,c2)
    #
    #                     if clusterDist <= minAvg:
    #                         minAvg = clusterDist
    #                         bestChoice = (i,i+1+j)
    #
    #             # if there is no merging to be done
    #             if bestChoice is None:
    #                 return clusters
    #
    #             i,j = bestChoice
    #             assert(j > i)
    #             c2 = clusters.pop(j)
    #             c1 = clusters.pop(i)
    #             clusters.append([c1,c2])
    #
    #     return clusters

    def __fit__(self,markings,user_ids,dbscan_preprocess=False,fname=None):
        #if len(markings) > 500:
        #    raise TooBig()
        print "Number of markings: " + str(len(markings))
        markings = [(x,y+random.uniform(0,0.0001)) for (x,y) in markings]

        # X = np.array(XYpts)
        # #use DBSCAN to create connectivity constraints
        # #keep expanding until there is no more noise
        # for first_epsilon in [25,50,100,200,300,400,600,800]:
        #     db = DBSCAN(eps=first_epsilon, min_samples=min(3,len(XYpts))).fit(X)
        #
        #     if not(-1 in db.labels_):
        #         break
        end_clusters =[]
        start = time.time()
        if dbscan_preprocess:
            pass

            assert(not(-1 in db.labels_))
            labels = db.labels_
        else:
            #starting_clusters = [((x,y),(x,y,user)) for (x,y), user in zip(markings,user_ids)]
            starting_clusters = [AgglomerativeLeaf(pt,user) for pt,user in zip(markings,user_ids)]
            self.__set_distances__(starting_clusters)
            end_clusters = self.__agglomerate_clusters__(starting_clusters,fname)
        # end_clusters = []
        # #print max(labels)
        # #do agglomerative clustering on each individual "sub" cluser
        # for k in sorted(set(labels)):
        #     clusters = [(x,y,user) for (x,y), user, l in zip(XYpts,user_ids,labels) if l == k]
        #     #keep on merging as much as possible
        #     while len(clusters) > 1:
        #         minAvg = float("inf")
        #         bestChoice = None
        #
        #         for i,c1 in enumerate(clusters):
        #             for j,c2 in enumerate(clusters[i+1:]):
        #                 clusterDist = self.averageLinkage(c1,c2)
        #
        #                 if clusterDist <= minAvg:
        #                     minAvg = clusterDist
        #                     bestChoice = (i,i+1+j)
        #
        #         i,j = bestChoice
        #         assert(j > i)
        #         c2 = clusters.pop(j)
        #         c1 = clusters.pop(i)
        #         clusters.append([c1,c2])
        #
        #     end_clusters.append(deepcopy(clusters[0]))
        #
        # print "+ " + str(len(end_clusters))
        #
        # #now merge each of these "intermediate" clusters together
        # #stop if the minAverage is infinity - at this point we are just merging pts from the same users
        # while len(end_clusters) > 1:
        #     minAvg = float("inf")
        #     bestChoice = None
        #     for i,c1 in enumerate(end_clusters):
        #         for j,c2 in enumerate(end_clusters[i+1:]):
        #             clusterDist = self.averageLinkage(c1,c2)
        #
        #             if clusterDist <= minAvg:
        #                 minAvg = clusterDist
        #                 bestChoice = (i,i+1+j)
        #
        #     #if minAvg == float("inf"):
        #     #    break
        #
        #     i,j = bestChoice
        #     assert(j > i)
        #     c2 = end_clusters.pop(j)
        #     c1 = end_clusters.pop(i)
        #     end_clusters.append([c1,c2])
        #
        # #print len(end_clusters[0])
        # #print len(end_clusters[0][0])
        # #print len(end_clusters[0][1])
        #
        # # newick = toNewick(end_clusters[0]) + ";"
        # # print newick
        # # t = Tree(newick)
        # # ts = TreeStyle()
        # # ts.show_leaf_name = True
        # # #ts.show_branch_length = True
        # # ts.show_branch_support = True
        # # t.render("/home/greg/mytree.pdf", w=400, units="mm",tree_style=ts)
        # # #t.show(tree_style=ts)
        results = [node.__get_cluster__() for node in end_clusters]
        markings,users = zip(*results)
        centers = []
        for cluster in markings:
            X,Y = zip(*cluster)
            centers.append((np.mean(X),np.mean(Y)))
        end = time.time()
        print "Seconds to cluster: " + str(end-start)
        print "Number of clusters: " + str(len(centers))
        return (centers, markings,users), end-start

class Ward(Agglomerative):
    def __init__(self):
        Agglomerative.__init__(self)

    def alpha(self,i,j,k):
        return (i+k)/float(i+j+k)

    def beta(self,i,j,k):
        return -k/float(i+j+k)

    def gamma(self):
        return 0