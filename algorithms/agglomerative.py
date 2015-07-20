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

# class AbstractNode:
#     def __init__(self):
#         self.value = None
#         self.rchild = None
#         self.lchild = None
#
#         self.parent = None
#         self.depth = None
#         self.height = None
#
#         self.users = None
#
#     def __set_parent__(self,node):
#         assert isinstance(node,InnerNode)
#         self.parent = node
#
#     @abc.abstractmethod
#     def __traversal__(self):
#         return []
#
#     def __set_depth__(self,depth):
#         self.depth = depth
#
#
# class LeafNode(AbstractNode):
#     def __init__(self,value,index,user=None):
#         AbstractNode.__init__(self)
#         self.value = value
#         self.index = index
#         self.users = [user,]
#         self.height = 0
#         self.pts = [value,]
#
#     def __traversal__(self):
#         return [(self.value,self.index),]
#
# class InnerNode(AbstractNode):
#     def __init__(self,rchild,lchild,dist=None):
#         AbstractNode.__init__(self)
#         assert isinstance(rchild,(LeafNode,InnerNode))
#         assert isinstance(lchild,(LeafNode,InnerNode))
#
#         self.rchild = rchild
#         self.lchild = lchild
#
#         rchild.__set_parent__(self)
#         lchild.__set_parent__(self)
#
#         self.dist = dist
#
#         assert (self.lchild.users is None) == (self.rchild.users is None)
#         if self.lchild.users is not None:
#             self.users = self.lchild.users[:]
#             self.users.extend(self.rchild.users)
#
#         self.pts = self.lchild.pts[:]
#         self.pts.extend(self.rchild.pts[:])
#
#         self.height = max(rchild.height,lchild.height)+1
#
#     def __traversal__(self):
#         retval = self.rchild.__traversal__()
#         retval.extend(self.lchild.__traversal__())
#
#         return retval

class Agglomerative(clustering.Cluster):
    def __init__(self,shape):
        clustering.Cluster.__init__(self,shape)
        self.all_distances = []
        self.max = 0

        self.correction_alg = multiClickCorrect.MultiClickCorrect(overlap_threshold=1,min_cluster_size=2,dist_threshold=50)

    # def __add_cluster(self,cluster_centers,end_clusters,end_users,node):
    #     # if len(node.pts) < 4:
    #     #     cluster_centers.append(None)
    #     #     end_clusters.append(None)
    #     #     end_users.append(None)
    #     # else:
    #     cluster_centers.append([np.median(axis) for axis in zip(*node.pts)])
    #     end_clusters.append(node.pts)
    #     end_users.append(node.users)
    #
    #     return cluster_centers,end_clusters,end_users

    # def __results_to_json__(self,node):
    #     results = {}
    #     results["center"] = [np.median(axis) for axis in zip(*node.pts)]
    #     results["points"] = node.pts
    #     results["users"] = node.users
    #
    #     return results

    def __inner_fit__(self,markings,user_ids,fname=None):
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
        # todo - find if there is a better way than just accepting markings from at most 15 users
        # useable_users = list(set(user_ids))[:15]
        # all_combined = [(m,u,t) for m,u,t in zip(markings,user_ids,tools) if u in useable_users]
        markings = markings[:15]
        user_ids = user_ids[:15]

        # in this special case, all of the markings are from different users - so we have only one cluster
        if len(user_ids) == len(set(user_ids)):
            result = {"users":user_ids,"points":markings}
            result["center"] = [np.median(axis) for axis in zip(*result["points"])]
            return [result],0

        # markings,user_ids,tools = zip(*all_combined)

        assert len(markings) == len(user_ids)
        assert isinstance(user_ids,tuple)
        user_ids = list(user_ids)
        start = time.time()

        all_users = set()

        # this converts stuff into panda format - probably a better way to do this but the labels do seem
        # necessary
        labels = [str(i) for i in markings]
        param_labels = [str(i) for i in range(len(markings[0]))]
        df = pd.DataFrame(np.array(markings), columns=param_labels, index=labels)
        row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
        # use ward metric to do the actual clustering
        row_clusters = linkage(row_dist, method='ward')

        # use the results to build a tree representation
        # nodes = [LeafNode(pt,ii,user=user) for ii,(user,pt) in enumerate(zip(user_ids,markings))]
        results = [{"users":[u],"points":[p]} for u,p in zip(user_ids,markings)]

        # read through the results
        # each row gives a cluster/node to merge
        # if one any two clusters have a user in common - don't merge them - and represent this by a None cluster
        # if trying to merge with a None cluster - this gives a None cluster as well
        for merge in row_clusters:
            rchild_index = int(merge[0])
            lchild_index = int(merge[1])
            # print len(nodes),rchild_index,lchild_index

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
                lnode["center"] = [np.median(axis) for axis in zip(*lnode["points"])]
                results.append(None)
            # maybe just the lnode is done:
            elif (lnode is None) or ("center" in lnode):
                rnode["center"] = [np.median(axis) for axis in zip(*rnode["points"])]
                results.append(None)
            else:
                # check if we should merge - only if there is no overlap
                # otherwise "cap" or terminate both nodes
                intersection = [u for u in rnode["users"] if u in lnode["users"]]

                # if there are users in common, add to the end clusters list (which consists of cluster centers
                # the points in each cluster and the list of users)
                if intersection != []:
                    rnode["center"] = [np.median(axis) for axis in zip(*rnode["points"])]
                    lnode["center"] = [np.median(axis) for axis in zip(*lnode["points"])]
                    results.append(None)
                else:
                    # else just merge
                    users = rnode["users"]
                    points = rnode["points"]
                    users.extend(lnode["users"])
                    points.extend(lnode["points"])
                    results.append({"users":users,"points":points})

        # go and remove all non terminal nodes from the results
        print
        print results[-1]
        if results[-1] is not None:
            print sorted(results[-1]["users"])
            print sorted(user_ids)
            print len(user_ids),len(set(user_ids))
        found_cluster = False
        for i in range(len(results)-1,-1,-1):
            if (results[i] is None) or ("center" not in results[i]):
                results.pop(i)
            else:
                found_cluster = True

        assert found_cluster

        end = time.time()
        # print [len(r["users"]) for r in results]
        # results = self.correction_alg.__fix__(results)

        print results
        return results,end-start

    def __check__(self):
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