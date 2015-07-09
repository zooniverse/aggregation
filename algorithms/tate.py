#!/usr/bin/env python
__author__ = 'greg'
from panoptes_api import PanoptesAPI,InvalidMarking
import agglomerative
import clustering
import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
import abc
import re
import random
import unicodedata
import os
from copy import deepcopy
import itertools

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

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

def Levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

class TextCluster(clustering.Cluster):
    def __init__(self,project_api,min_cluster_size=1,mapping=None):
        clustering.Cluster.__init__(self,project_api,min_cluster_size)

    def __get_aggregation_lines__(self,lines):
        rep = {" ":"@","-":"#","(":"{",">":"^"}
        rep = dict((re.escape(k), v) for k, v in rep.iteritems())
        pattern = re.compile("|".join(rep.keys()))

        id_ = str(random.uniform(0,1))

        with open(base_directory+"/Databases/transcribe"+id_+".fasta","wb") as f:
           for line in lines:
                if isinstance(line,tuple):
                    # we have a list of text segments which we should join together
                    line = "".join(line)



                # line = unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                assert isinstance(line,str)
                fasta_line = pattern.sub(lambda m: rep[re.escape(m.group(0))], line)

                # for i in range(max_length-len(line)):
                #     fasta_line += "-"

                try:
                    f.write(">\n"+fasta_line+"\n")
                except UnicodeEncodeError:
                    print fasta_line
                    print unicodedata.normalize('NFKD', fasta_line).encode('ascii','ignore')
                    raise

        t = "mafft --anysymbol " + base_directory+"/Databases/transcribe"+id_+".fasta>"+base_directory+"/Databases/transcribe"+id_+".out 2> /dev/null"
        os.system(t)

        aligned_text = []
        with open(base_directory+"/Databases/transcribe"+id_+".out","rb") as f:
            cumulative_line = ""
            # print "here here"
            for line in f.readlines():
                # print line[:-1]
                if (line == ">\n"):
                    if (cumulative_line != ""):
                        aligned_text.append(cumulative_line)
                        cumulative_line = ""
                else:
                    cumulative_line += line[:-1]

            aligned_text.append(cumulative_line)

        os.remove(base_directory+"/Databases/transcribe"+id_+".fasta")
        os.remove(base_directory+"/Databases/transcribe"+id_+".out")

        return aligned_text

    def __accuracy__(self,s):
        assert isinstance(s,str)
        assert len(s) > 0
        return sum([1 for c in s if c != "-"])/float(len(s))

    def __agreement__(self,text):
        assert isinstance(text,list)
        assert len(text) > 1
        assert isinstance(text[0],str)
        assert min([len(t) for t in text]) == max([len(t) for t in text])

        retval = []

        for t in text:
            leftmost_char = -1
            rightmost_char = -1
            for i,c in enumerate(t):
                if c != "-":
                    leftmost_char = i
                    break
            for i,c in reversed(list(enumerate(t))):
                if c != "-":
                    rightmost_char = i
                    break

            agreement = 0

            for i in range(leftmost_char,rightmost_char+1):
                c = [t2[i].lower() for t2 in text]
                if min(c) == max(c):
                    assert c[0] != "-"
                    agreement += 1

            retval.append(agreement/float(rightmost_char-leftmost_char+1))
        return retval

    def __complete_agreement__(self,text):
        assert isinstance(text,list)
        assert len(text) > 1
        assert isinstance(text[0],str)
        assert min([len(t) for t in text]) == max([len(t) for t in text])

        agreement = 0
        # print text
        for i in range(len(text[0])):

            c = [t[i].lower() for t in text if t[i] != "-"]
            if min(c) == max(c):
                assert c[0] != "-"
                agreement += 1

        return agreement/float(len(text[0]))

    def __inner_fit__(self,markings,user_ids,tools,fname=None):
        M = []
        texts = []

        for x1,x2,y1,y2,text in markings:
            dist = (x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)
            texts.append(text)
            try:
                tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
                theta = math.atan(tan_theta)
            except ZeroDivisionError:
                theta = math.pi/2.

            M.append((dist,theta))
            # print str(dist) + "\t" + str(theta) + "\t" + text

        # make sure all of the points are distinct
        if len(M) != len(set(M)):
            print user_ids
            print markings

        assert len(M) == len(set(M))

        ordering  = self.__fit2__(M,user_ids)

        current_lines = {}
        clusters = []

        for a,b in ordering:
            # a - line values - "intercept" and slope
            i = M.index(a)
            user = user_ids[i]
            text = texts[i]

            if "\n" in text:
                print "multiline - skipping"
                continue

            if text == "":
                print "empty line"
                continue

            # convert from unicode to ascii
            assert isinstance(text,unicode)
            text = text.encode('ascii','ignore')

            #  # todo - can this be done better?
            text = re.sub("\[deletion\].*\[/deletion\]","",text)
            text = re.sub(r'\[deletion\].*\[\\deletion\]',"",text)
            text = re.sub("\[illegible\].*\[/illegible\]","",text)
            text = re.sub(r'\[deletionhas\]\[/deletion\]',"",text)
            text = re.sub(r'\[has\]',"",text)
            text = re.sub(r'\(deleted\)',"",text)
            text = re.sub(r'\[deletion\]',"",text)

            # if we have an empty cluster, just add the line
            if current_lines == {}:
                current_lines = {user:text}
            else:
                # need to see if we want to merge
                # do we already have some text from this user for this current cluster?
                # if so, should we merge the text or start a new cluster?
                if user in current_lines:
                    # we have three possibilities - go with the current text, new text or a combination of the two
                    if len(current_lines) > 1:
                        # current accuracy
                        # ordering is important here
                        lines_and_users = sorted(current_lines.items(),key = lambda x:x[0])
                        sorted_users,sorted_lines = zip(*lines_and_users)
                        user_index = sorted_users.index(user)

                        # start with the base accuracy - i.e. the current one
                        aligned_text = self.__get_aggregation_lines__(sorted_lines)
                        current_accuracy = self.__agreement__(aligned_text)

                        # if we do decide to go with the new text in a new cluster
                        text_for_current_cluster = sorted_lines[user_index]
                        text_for_new_cluster = text
                        max_accuracy = current_accuracy[user_index]

                        new_lines = list(sorted_lines)
                        # start by trying straight up replacing
                        new_lines[user_index] = text
                        new_aligned = self.__get_aggregation_lines__(new_lines)
                        new_accuracy = self.__agreement__(new_aligned)

                        if new_accuracy[user_index] > max_accuracy:
                            text_for_current_cluster = text
                            text_for_new_cluster = sorted_lines[user_index]
                            max_accuracy = new_accuracy[user_index]


                        # now try merging the text - we don't know the order we are supposed to concat them in
                        # todo - figure out a better way than just trying all possibilities
                        # if a list - we are already dealing with text segments
                        if isinstance(sorted_users[user_index],list):
                            text_segments = sorted_lines[user_index]
                        else:
                            assert isinstance(sorted_lines[user_index],str)
                            text_segments = [sorted_lines[user_index],]

                        text_segments.append(text)
                        for l in  itertools.permutations(text_segments):
                            new_lines = list(sorted_lines)
                            new_lines[user_index] = l
                            # print l
                            new_aligned = self.__get_aggregation_lines__(new_lines)
                            new_accuracy = self.__agreement__(new_aligned)
                            if new_accuracy[user_index] > max_accuracy:
                                text_for_current_cluster = l
                                text_for_new_cluster = None
                                max_accuracy = new_accuracy[user_index]

                            # todo - figure out how to handle this case
                            # assert current_accuracy[user_index] > new_accuracy[user_index]

                        current_lines[user] = text_for_current_cluster
                        if text_for_new_cluster is not None:
                            clusters.append(current_lines.values())
                            current_lines = {user:text_for_new_cluster}

                    else:
                        # if len(current_lines) == 1, then the current user is the only user with text in the cluster
                        # so far, so really not sure if we should try merging or creating a new cluster
                        assert False

                else:
                    # does adding this line to the cluster make sense?
                    # compare against the current accuracy - if we only have 1 line so far,
                    # current accuracy is NA
                    lines_and_users = sorted(current_lines.items(),key = lambda x:x[0])
                    sorted_users,sorted_lines = zip(*lines_and_users)
                    if len(current_lines) > 1:
                        aligned_text = self.__get_aggregation_lines__(sorted_lines)
                        current_accuracy = self.__agreement__(aligned_text)
                    else:
                        current_accuracy = -1

                    # what would the accuracy be if we added in this new user's line?
                    new_lines = list(sorted_lines)
                    assert isinstance(sorted_users,tuple)
                    # user_index = sorted_users.index(user)

                    # start by trying straight up replacing
                    new_lines.append(text)
                    new_aligned = self.__get_aggregation_lines__(new_lines)
                    new_accuracy = self.__agreement__(new_aligned)

                    if min(new_accuracy) >= 0.6:
                        current_lines[user] = text
                    else:
                        clusters.append(current_lines.values())
                        current_lines = {user:text}



        clusters.append(current_lines.values())

        # remove any clusters which have only one user
        for cluster_index in range(len(clusters)-1,-1,-1):
            if len(clusters[cluster_index]) == 1:
                clusters.pop(cluster_index)

        # after removing such "error" clusters there may be adjacent clusters which should be merged
        for cluster_index in range(len(clusters)-2,-1,-1):
            t_cluster = clusters[cluster_index][:]
            t_cluster.extend(clusters[cluster_index+1])

            aligned_text = self.__get_aggregation_lines__(t_cluster)

            accuracy = self.__agreement__(aligned_text)
            if min(accuracy)> 0.6:
                clusters[cluster_index].extend(clusters.pop(cluster_index+1))

        for c in clusters:
            for l in c:
                print l

            print
        assert False

        return ((),(),()),0

    def __fit2__(self,markings,user_ids,jpeg_file=None,debug=False):
        # l = [[(u,m[0]) for m in marking] for u,marking in zip(user_ids,markings)]
        user_list,pts_list = user_ids,markings
        # assert len(pts_list) == len(list(set(pts_list)))
        labels = range(len(pts_list))
        variables = ["X","Y"]
        # X = np.random.random_sample([5,3])*10
        df = pd.DataFrame(list(pts_list),columns=variables, index=labels)

        row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)

        row_clusters = linkage(row_dist, method='single')

        nodes = [LeafNode(pt,ii) for ii,pt in enumerate(pts_list)]

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
        return reachability_ordering
        # assert False
        # reachability_distance = [float("inf"),]
        # for ii, leaf in enumerate(reachability_ordering[1:]):
        #     # print reachability_ordering[ii]
        #     # print reachability_ordering[ii+1]
        #     node1 = nodes[reachability_ordering[ii][1]]
        #     node2 = nodes[reachability_ordering[ii+1][1]]
        #
        #     reachability_distance.append(lowest_common_ancestor(node1,node2))
        #
        #
        #
        # # find the "important" local maxima
        # important_local_maxima = []
        # for i in range(1,len(reachability_distance)-1):
        #     dist = reachability_distance[i]
        #     other_distances = []
        #     if i > 0:
        #         other_distances.append(reachability_distance[i-1])
        #     if i < (len(reachability_distance)-1):
        #         other_distances.append(reachability_distance[i+1])
        #     if dist > max(other_distances):
        #         if np.mean(other_distances) < 0.8*dist:
        #             important_local_maxima.append((i,dist))
        #
        # clusters = create_clusters(zip(*reachability_ordering)[0],important_local_maxima)
        # users_per_cluster = [[user_list[pts_list.index(p)] for p in c] for c in clusters]
        # cluster_centers = [[np.mean(axis) for axis in zip(*c)] for c in clusters]
        # return (cluster_centers,clusters,users_per_cluster),0

def text_mapping(marking,image_dimensions):
    # want to extract the params x1,x2,y1,y2 but
    # ALSO make sure that x1 <= x2 and flip if necessary

    x1 = marking["startPoint"]["x"]
    x2 = marking["endPoint"]["x"]
    y1 = marking["startPoint"]["y"]
    y2 = marking["endPoint"]["y"]

    try:
        text = marking["text"]
    except KeyError:
        raise InvalidMarking(marking)

    if x1 <= x2:
        return x1,x2,y1,y2,text
    else:
        return x2,x1,y2,y1,text

def text_mapping2(marking,image_dimensions):
    x1 = marking["x1"]
    x2 = marking["x2"]
    y2 = marking["y2"]
    y1 = marking["y1"]

    try:
        text = marking["text"]
    except KeyError:
        raise InvalidMarking(marking)

    if x1 <= x2:
        return x1,x2,y1,y2,text
    else:
        return x2,x1,y2,y1,text

class Tate(PanoptesAPI):
    def __init__(self):
        PanoptesAPI.__init__(self,"tate")

        self.marking_params_per_shape = dict()
        self.marking_params_per_shape["text"] = text_mapping2

        self.workflows[683] = {},{"init":["text"]}

    def __readin_tasks__(self,workflow_id):
        marking_tasks = {"T2":["text"]}
        classification_tasks = {}

        return classification_tasks,marking_tasks

    # def __task_setup__(self):
    #     self.classification_tasks = {}
    #     self.task_type["init"] = "drawing"
    #     self.task_type["T1"] = None
    #     self.shapes_per_tool["init"] = []
    #     self.shapes_per_tool["init"].append("text")

bentham = [4150,4151,4152,4153,4154]
tate = [4127,4129,4130,4131,4132,4133,4136]
project = Tate()
# project.__migrate__()
project.__set_clustering_alg__({"text":(TextCluster,{})})
project.__aggregate__(workflows=[683])

# project.__set_subjects__([4150])
# project.__set_clustering_alg__(TextCluster)
# project.__cluster__()
# project.__postgres_backup__()