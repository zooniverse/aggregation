__author__ = 'ggdhines'
import clustering
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
import time
import abc
from scipy.stats import beta
import math
import numpy
import multiClickCorrect
import json
import random
from copy import deepcopy


def text_line_mappings(line_segments):
    """
    use if we want to cluster based on Hesse normal form - but want to retain the original values
    :param line_segment:
    :return:
    """
    reduced_markings = []

    for line_seg in line_segments:
        x1,y1,x2,y2,text = line_seg

        x2 += random.uniform(-0.0001,0.0001)
        x1 += random.uniform(-0.0001,0.0001)

        dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)

        try:
            tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
            theta = math.atan(tan_theta)
        except ZeroDivisionError:
            theta = math.pi/2.

        reduced_markings.append((dist,theta,text))

    return reduced_markings


class Agglomerative(clustering.Cluster):
    def __init__(self,shape,project,additional_params):
        clustering.Cluster.__init__(self,shape,None,additional_params)
        self.all_distances = []
        self.max = 0

        self.correction_alg = multiClickCorrect.MultiClickCorrect(overlap_threshold=1,min_cluster_size=2,dist_threshold=20)

    def __agglomerative__(self,markings):
        """
        runs an initial agglomerative clustering over the given markings
        :param markings:
        :return:
        """
        # this converts stuff into panda format - probably a better way to do this but the labels do seem
        # necessary
        labels = [str(i) for i in markings]
        param_labels = [str(i) for i in range(len(markings[0]))]

        df = pd.DataFrame(np.array(markings), columns=param_labels, index=labels)
        row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
        # use ward metric to do the actual clustering
        row_clusters = linkage(row_dist, method='ward')

        return row_clusters

    def __merge_clusters__(self,cluster1,cluster2):
        """
        merge two clusters - which ideally should not have any users in common
        that should already have been checked
        :param cluster1:
        :param cluster2:
        :return:
        """
        new_cluster = deepcopy(cluster1)
        new_cluster["users"].extend(cluster2["users"])
        new_cluster["cluster members"].extend(cluster2["cluster members"])
        new_cluster["tools"].extend(cluster2["tools"])
        new_cluster["num users"] += cluster2["num users"]

        return new_cluster

    def __tree_traverse__(self,dendrogram,markings,user_ids,tools):
        """
        given a list representation of a dendrogram - resulting from running agglomerative clustering
        https://en.wikipedia.org/wiki/Dendrogram\
        each node in the tree is a dictionary containing things like "cluster members"
        cap subtrees - that is find subtrees that should not be merged since they contain common users
        so if A and B are siblings in the dendrogram and have common users, the parent of A and B
        will be set to null. Also to indicate that A and B are "final clusters"   add "center" to those values
        finally return a list of only the final clusters
        :param dendrogram: a list representation of the results (tree) from agglomerative clustering
        :param markings: the raw markings
        :param user_ids: the user ids per raw markings
        :param tools: the tools associated with the raw markings
        :return:
        """
        # todo - why do I have tool_classification in here?
        results = [{"users":[u],"cluster members":[p],"tools":[t],"num users":1} for u,p,t in zip(user_ids,markings,tools)]
        # cluster_mergers is a list representation of a tree
        # let's traverse this tree looking for mergers between clusters with common users - those clusters should
        # not be merged. (we'll call those two child clusters "capped clusters"
        # if trying to merge with a None cluster - this gives a None cluster as well - since these are above capped clusters
        for merge in dendrogram:
            rchild_index = int(merge[0])
            lchild_index = int(merge[1])

            rnode = results[rchild_index]
            lnode = results[lchild_index]

            # use "center" being in a dict to indicate that a node shouldn't be merged any more
            # use None to indicate that a "child" of the current node was a "terminal" node
            # if either node/cluster is None, we have already encountered overlapping users
            # if both are already terminal - just append None
            if ((rnode is None) or ("center" in rnode)) and ((lnode is None) or ("center" in lnode)):
                results.append(None)
            # maybe just the rnode was is already done - in which case "finish" the lnode
            elif (rnode is None) or ("center" in rnode):
                lnode["center"] = [np.median(axis) for axis in zip(*lnode["cluster members"])]
                results.append(None)
            # maybe just the lnode is done:
            elif (lnode is None) or ("center" in lnode):
                rnode["center"] = [np.median(axis) for axis in zip(*rnode["cluster members"])]
                results.append(None)
            else:
                # check if we should merge - only if there is no overlap
                # otherwise "cap" or terminate both nodes
                intersection = [u for u in rnode["users"] if u in lnode["users"]]
                assert "center" not in rnode
                assert "center" not in lnode

                # if there are users in common, add to the end clusters list (which consists of cluster centers
                # the points in each cluster and the list of users)
                if intersection != []:
                    rnode["center"] = [np.median(axis) for axis in zip(*rnode["cluster members"])]
                    lnode["center"] = [np.median(axis) for axis in zip(*lnode["cluster members"])]
                    results.append(None)
                else:
                    # else just merge
                    merged_clusters = self.__merge_clusters__(rnode,lnode)
                    # print(merged_clusters.keys())
                    results.append(merged_clusters)

        # go and remove all non terminal nodes from the results
        for i in range(len(results)-1,-1,-1):
            # if None => corresponds to being above a terminal node
            # or center not in results => corresponds to being beneath a terminal node
            # in both cases does not mean immediately above or below
            if (results[i] is None) or ("center" not in results[i]):
                results.pop(i)
        return results

    def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions,subject_id):
        """
        the actual clustering algorithm
        markings and user_ids should be the same length - a one to one mapping
        :param markings:
        :param user_ids:
        :param jpeg_file:
        :param debug:
        :param gold_standard:
        :param subject_id:
        :return:
        """
        assert len(markings) == len(user_ids)
        assert len(markings) == len(reduced_markings)

        if isinstance(user_ids,tuple):
            user_ids = list(user_ids)
        assert isinstance(user_ids,list)
        start = time.time()

        if len(user_ids) == len(set(user_ids)):
            # all of the markings are from different users => so only one cluster
            result = {"users":user_ids,"cluster members":markings,"tools":tools,"num users":len(user_ids)}
            result["center"] = [np.median(axis) for axis in zip(*markings)]
            return [result],0

        # cluster based on the reduced markings, but list the clusters based on their original values
        dendrogram = self.__agglomerative__(reduced_markings)

        results = self.__tree_traverse__(dendrogram,markings,user_ids,tools)

        # todo - this is just for debugging
        for j in results:
            assert "num users" in j

        end = time.time()
        return results,end-start

    def __check__(self):
        """
        purely for debugging and dev - trying to understand the distribution of points in a cluster
        :return:
        """
        self.all_distances = [d/max(self.all_distances) for d in self.all_distances]
        mean=numpy.mean(self.all_distances)
        var=numpy.var(self.all_distances,ddof=1)

        ii = len(self.all_distances)
        while var >= (mean*(1-mean)):
            ii -= 1
            mean=numpy.mean(self.all_distances[:ii])
            var=numpy.var(self.all_distances[:ii],ddof=1)

        alpha1=mean*(mean*(1-mean)/var-1)
        beta1=alpha1*(1-mean)/mean
        for d in sorted(self.all_distances):
            print d,beta.cdf(d,alpha1,beta1)