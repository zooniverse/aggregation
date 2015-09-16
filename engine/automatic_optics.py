__author__ = 'greg'
from clustering import Cluster
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
import abc
import math

class AbstractNode:
    def __init__(self):
        self.value = None
        self.rchild = None
        self.lchild = None

        self.parent = None
        self.depth = None
        self.height = None

        self.users = None

    def __set_parent__(self,node):
        assert isinstance(node,InnerNode)
        self.parent = node

    @abc.abstractmethod
    def __traversal__(self):
        return []

    def __set_depth__(self,depth):
        self.depth = depth


class LeafNode(AbstractNode):
    def __init__(self,value,index,user=None):
        AbstractNode.__init__(self)
        self.value = value
        self.index = index
        self.users = [user,]
        self.height = 0
        self.pts = [value,]

    def __traversal__(self):
        return [(self.value,self.index),]

class InnerNode(AbstractNode):
    def __init__(self,rchild,lchild,dist=None):
        AbstractNode.__init__(self)
        assert isinstance(rchild,(LeafNode,InnerNode))
        assert isinstance(lchild,(LeafNode,InnerNode))

        self.rchild = rchild
        self.lchild = lchild

        rchild.__set_parent__(self)
        lchild.__set_parent__(self)

        self.dist = dist

        assert (self.lchild.users is None) == (self.rchild.users is None)
        if self.lchild.users is not None:
            self.users = self.lchild.users[:]
            self.users.extend(self.rchild.users)

        self.pts = self.lchild.pts[:]
        self.pts.extend(self.rchild.pts[:])

        self.height = max(rchild.height,lchild.height)+1

    def __traversal__(self):
        retval = self.rchild.__traversal__()
        retval.extend(self.lchild.__traversal__())

        return retval


def set_depth(node,depth=0):
    assert isinstance(node,AbstractNode)
    node.__set_depth__(depth)

    if node.rchild is not None:
        set_depth(node.rchild,depth+1)
    if node.lchild is not None:
        set_depth(node.lchild,depth+1)


def lowest_common_ancestor(node1,node2):
    assert isinstance(node1,LeafNode)
    assert isinstance(node2,LeafNode)

    depth1 = node1.depth
    depth2 = node2.depth

    # make sure that the first node is the "shallower" node
    if depth1 > depth2:
        temp = node2
        node2 = node1
        node1 = temp

        depth1 = node1.depth
        depth2 = node2.depth
    while depth2 > depth1:
        node2 = node2.parent
        depth2 = node2.depth

    while node1 != node2:
        node1 = node1.parent
        node2 = node2.parent

    return node1.height

def create_clusters(ordering,maxima):
    if maxima == []:
        return [ordering,]

    next_maxima = max(maxima,key=lambda x:x[1])

    split = next_maxima[0]
    left_split = ordering[:split]
    right_split = ordering[split:]

    maxima_index = maxima.index(next_maxima)
    left_maximia = maxima[:maxima_index]
    right_maximia = maxima[maxima_index+1:]
    # print right_maximia
    # need to adjust the indices for the right hand values
    right_maximia = [(i-split,j) for (i,j) in right_maximia]
    # print right_maximia

    retval = create_clusters(left_split,left_maximia)
    retval.extend(create_clusters(right_split,right_maximia))

    return retval


class AutomaticOptics(Cluster):
    def __init__(self, project_api,min_cluster_size=1):
        Cluster.__init__(self, project_api,min_cluster_size)

    # def __correct__(self,subject_id):
    #     """
    #     find any nearest neighbour tuples of clusters which have no users in common and merge them
    #     :return:
    #     """
    #     results = self.clusterResults[subject_id]
    #     i = 0
    #     # the length of results[2] may and probably will change as we correct things
    #     # so don't use a for loop
    #     # -1 so we always have at least one more element to compare against
    #
    #     while i < len(results[2])-1:
    #         users_i = results[2][i]
    #         pts_i = results[1][i]
    #         cluster_i = results[0][i]
    #
    #         closest_distance = float("inf")
    #         closest_neighbour = None
    #         overlap = None
    #         # check the overlap between i and all clusters "above" it - overlap is symmetrical so we don't need
    #         # to check both ways. Also we are going backwards so that we can pop stuff from the list without
    #         # messing the indexing up
    #         for j in range(len(results[2])-1,i,-1):
    #             assert j != i
    #             users_j = results[2][j]
    #             cluster_j = results[0][j]
    #             dist = math.sqrt(sum([(pi-pj)**2 for (pi,pj) in zip(cluster_i,cluster_j)]))
    #
    #             if dist < closest_distance:
    #                 closest_distance = dist
    #                 overlap = [u for u in users_j if u in users_i]
    #                 closest_neighbour = j
    #
    #         if len(overlap) == 0:
    #             # remove the j'th element and merge it with the i'th one
    #             center = results[0].pop(closest_neighbour)
    #             pts = results[1].pop(closest_neighbour)
    #             users = results[2].pop(closest_neighbour)
    #
    #             # to allow for generalizations where the overlap is non-empty, we need  a way to merge points
    #             for users in overlap:
    #                 # todo: do generalization
    #                 pass
    #
    #             # todo: find a better way to do this, probably stop it from being a tuple in the first place
    #             results[1][i] = list(results[1][i])
    #             results[1][i].extend(pts)
    #             results[2][i].extend(users)
    #
    #             # calculate the new center
    #             results[0][i] = [np.mean(axis) for axis in zip(*results[1][i])]
    #         # move on to the next element
    #         i += 1
    #
    #     print "ending length is " + str(len(results[2]))

    # def __fit__(self,markings,user_ids,jpeg_file=None,debug=False):
    def __inner_fit__(self,markings,user_ids,tools,reduced_markings):
        # print len(user_ids)
        # print len(markings)
        # l = [[(u,m) for m in marking] for u,marking in zip(user_ids,markings)]
        # user_list,pts_list = zip(*[item for sublist in l for item in sublist])
        # assert len(pts_list) == len(list(set(pts_list)))
        labels = range(len(markings))
        variables = ["X","Y"]
        # X = np.random.random_sample([5,3])*10
        df = pd.DataFrame(list(markings),columns=variables, index=labels)

        row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)

        row_clusters = linkage(row_dist, method='single')

        nodes = [LeafNode(pt,ii) for ii,pt in enumerate(markings)]

        for merge in row_clusters:
            rchild_index = int(merge[0])
            lchild_index = int(merge[1])
            dist = float(merge[2])

            rnode = nodes[rchild_index]
            lnode = nodes[lchild_index]

            # if both nodes are leaf nodes, just merge them
            if isinstance(rnode,LeafNode) and isinstance(lnode,LeafNode):
                nodes.append(InnerNode(rnode,lnode,dist))
            # if rnode is an inner node - we might need to merge into it
            elif isinstance(lnode,LeafNode):
                r_dist = rnode.dist

                if r_dist == dist:
                    # merge
                    pass
                else:
                    # create a new parent node
                    nodes.append(InnerNode(rnode,lnode,dist))
            elif isinstance(rnode,LeafNode):
                l_dist = lnode.dist

                if l_dist == dist:
                    # merge
                    pass
                else:
                    # create a new parent node
                    nodes.append(InnerNode(rnode,lnode,dist))
            else:
                # we have two inner nodes
                l_dist = lnode.dist
                r_dist = rnode.dist

                if dist == l_dist:
                    assert dist == r_dist
                    assert False
                else:
                    nodes.append(InnerNode(rnode,lnode,dist))

        # set the depths of all of the nodes
        set_depth(nodes[-1])

        reachability_ordering = nodes[-1].__traversal__()
        reachability_distance = [float("inf"),]
        for ii, leaf in enumerate(reachability_ordering[1:]):
            # print reachability_ordering[ii]
            # print reachability_ordering[ii+1]
            node1 = nodes[reachability_ordering[ii][1]]
            node2 = nodes[reachability_ordering[ii+1][1]]

            reachability_distance.append(lowest_common_ancestor(node1,node2))

        # find the "important" local maxima
        important_local_maxima = []
        for i in range(1,len(reachability_distance)-1):
            dist = reachability_distance[i]
            other_distances = []
            if i > 0:
                other_distances.append(reachability_distance[i-1])
            if i < (len(reachability_distance)-1):
                other_distances.append(reachability_distance[i+1])
            if dist > max(other_distances):
                if np.mean(other_distances) < 0.75*dist:
                    important_local_maxima.append((i,dist))

        clusters = create_clusters(zip(*reachability_ordering)[0],important_local_maxima)
        users_per_cluster = [[user_ids[markings.index(p)] for p in c] for c in clusters]
        cluster_centers = [[np.mean(axis) for axis in zip(*c)] for c in clusters]
        results = []
        for centers,pts,users in zip(cluster_centers,clusters,users_per_cluster):
            results.append({"users":users,"cluster members":pts,"tools":[],"num users":len(users),"center":centers})

        return results,0