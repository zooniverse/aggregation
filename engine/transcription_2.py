#!/usr/bin/env python
__author__ = 'greg'
from aggregation_api import AggregationAPI,InvalidMarking
from classification import Classification
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
import requests
from aggregation_api import hesse_line_reduction
from scipy import spatial

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
    def __init__(self,shape,dim_reduction_alg):
        clustering.Cluster.__init__(self,shape,dim_reduction_alg)
        self.line_agreement = []

    def __get_aggregation_lines__(self,lines):
        # mafft doesn't deal well with some characters so need to replace them

        id_ = str(random.uniform(0,1))

        with open(base_directory+"/Databases/transcribe"+id_+".fasta","wb") as f:
           for line in lines:
                if isinstance(line,tuple):
                    # we have a list of text segments which we should join together
                    line = "".join(line)

                # line = unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                assert isinstance(line,str)

                # for i in range(max_length-len(line)):
                #     fasta_line += "-"

                try:
                    f.write(">\n"+line+"\n")
                except UnicodeEncodeError:
                    print line
                    print unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                    raise

        t = "mafft --text " + base_directory+"/Databases/transcribe"+id_+".fasta>"+base_directory+"/Databases/transcribe"+id_+".out 2> /dev/null"
        os.system(t)

        aligned_text = []
        with open(base_directory+"/Databases/transcribe"+id_+".out","rb") as f:
            cumulative_line = ""
            for line in f.readlines():
                if (line == ">\n"):
                    if (cumulative_line != ""):
                        aligned_text.append(cumulative_line)
                        cumulative_line = ""
                else:
                    cumulative_line += line[:-1]

            # todo - get this to work!!!
            # cumulative_line = undo_pattern.sub(lambda m: rep[re.escape(m.group(0))], cumulative_line)
            assert cumulative_line != ""
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
        for i in range(len(text[0])):

            c = [t[i].lower() for t in text if t[i] != "-"]
            if min(c) == max(c):
                assert c[0] != "-"
                agreement += 1

        return agreement/float(len(text[0]))

    def __inner_fit__(self,markings,user_ids,tools,reduced_markings):
        # we want to first cluster first just on dist and theta - ignoring the text contents
        # dist_list,theta_list,text_list,raw_pts_list = zip(*markings)
        # mapped_markings = zip(dist_list,theta_list)

        # cluster just on points, not on text
        dist_l,theta_l,text_l = zip(*reduced_markings)
        reduced_markings_without_text = zip(dist_l,theta_l)
        ordering  = self.__fit2__(reduced_markings_without_text,user_ids)

        # use the below 2 to build up each cluster
        current_lines = {}
        current_pts = {}
        clusters = []


        for a,b in ordering:
            # a - line values - "intercept" and slope
            user_index = reduced_markings_without_text.index(a)
            user = user_ids[user_index]
            # extract the corresponding text and the raw (unmapped) point
            # text = text_list[user_index]
            # raw_pt = raw_pts_list[user_index]

            text = markings[user_index][-1]
            raw_pt = markings[user_index][:-1]


            if "\n" in text:
                print "multiline - skipping"
                continue



            # convert from unicode to ascii
            assert isinstance(text,unicode)
            text = text.encode('ascii','ignore')
            # convert to lower case
            text = text.lower()

            # use capital letters to represent special characters
            # text = re.sub("\[deletion\].*\[/deletion\]","",text)
            # text = re.sub(r'\[deletion\].*\[\\deletion\]',"",text)
            # text = re.sub("\[illegible\].*\[/illegible\]","",text)
            # text = re.sub(r'\[deletionhas\]\[/deletion\]',"",text)
            # text = re.sub("\[insertion\].*\[/insertion\]","",text)
            # text = re.sub("\[underline\].*\[/underline\]","",text)
            # text = re.sub("\[notenglish\].*\[/notenglish\]","",text)
            # text = re.sub(r'\[has\]',"",text)
            # text = re.sub(r'\(deleted\)',"",text)
            # text = re.sub(r'\[deletion\]',"",text)
            # text = re.sub("\[insertion\]","",text)

            # deletion
            text = re.sub("\[deletion\]","A",text)
            text = re.sub("\[/deletion\]","B",text)
            # illegible
            text = re.sub("\[illegible\]","C",text)
            text = re.sub("\[/illegible\]","D",text)
            # insertion
            text = re.sub("\[insertion\]","E",text)
            text = re.sub("\[/insertion\]","F",text)
            # not english
            text = re.sub("\[notenglish\]","G",text)
            text = re.sub("\[/notenglish\]","H",text)
            # special characters for MAFFT
            text = re.sub(" ","I",text)
            text = re.sub("=","J",text)
            text = re.sub("\*","K",text)
            text = re.sub("\(","L",text)
            text = re.sub("\)","M",text)
            text = re.sub("<","N",text)
            text = re.sub(">","O",text)
            text = re.sub("-","P",text)



            # todo - find a way to fix this - stupid postgres/json
            text = re.sub(r'\'',"",text)

            # do this now, because all of the above subsitutions may have created an empty line
            if text == "":
                continue

            # if we have an empty cluster, just add the line
            if current_lines == {}:
                current_lines[user] = text #(text,special_characters)

                # adding the user id is slightly redundant but makes doing the actual clustering easier
                current_pts[user] = (raw_pt,user)
            else:
                # need to see if we want to merge
                # do we already have some text from this user for this current cluster?
                # IMPORTANT
                # VERY IMPORTANT
                # for the simplified transcription, we will assume that we should automatically start a new
                # cluster - i.e. we don't deal with split lines
                if user in current_pts:
                    clusters.append((current_lines.values(),current_pts.values()))
                    current_lines = {user:text} #(text,special_characters)}
                    current_pts = {user:(raw_pt,user)}
                else:
                    # does adding this line to the cluster make sense?
                    # compare against the current accuracy - if we only have 1 line so far,
                    # current accuracy is NA
                    users_and_lines = sorted(current_lines.items(),key = lambda x:x[0])
                    sorted_users,sorted_lines = zip(*users_and_lines)
                    # sorted_lines = zip(*sorted_pts)[-1]

                    # uncomment below if you want to compare the new accuracy against the old
                    # if len(current_lines) > 1:
                    #     aligned_text = self.__get_aggregation_lines__(sorted_lines)
                    #     current_accuracy = self.__agreement__(aligned_text)
                    # else:
                    #     current_accuracy = -1

                    # what would the accuracy be if we added in this new user's line?
                    new_lines = list(sorted_lines)
                    assert isinstance(sorted_users,tuple)
                    # user_index = sorted_users.index(user)

                    # start by trying straight up replacing
                    new_lines.append(text)
                    # print sorted_pts
                    # print new_lines
                    new_aligned = self.__get_aggregation_lines__(new_lines)
                    new_accuracy = self.__agreement__(new_aligned)

                    if min(new_accuracy) >= 0.6:
                        current_pts[user] = (raw_pt,user)
                        current_lines[user] = text
                    else:
                        clusters.append((current_lines.values(),current_pts.values()))
                        # current_pts = {user:(pt,text)}
                        current_lines = {user:text}
                        current_pts = {user:(raw_pt,user)}

        clusters.append((current_lines.values(),current_pts.values()))

        # remove any clusters which have only one user
        for cluster_index in range(len(clusters)-1,-1,-1):
            # print len(clusters[cluster_index][0])
            if len(clusters[cluster_index][0]) <= 1: #2
                # assert len(clusters[cluster_index][1]) == 1
                clusters.pop(cluster_index)

        if len(clusters) == 0:
            return [],0

        # if we have more than one cluster - some of them might need to be merged
        # after removing "error" cluster
        # to do so - revert back to Hesse format
        if len(clusters) > 1:

            hessen_lines = []

            for cluster_index in range(len(clusters)):
                lines_segments,users = zip(*clusters[cluster_index][1])
                x1_l, x2_l, y1_l, y2_l = zip(*lines_segments)
                x1,x2,y1,y2 = np.median(x1_l),np.median(x2_l),np.median(y1_l),np.median(y2_l)
                hessen_lines.append(hesse_line_reduction([[x1,x2,y1,y2],])[0])

            # print hessen_lines
            slope_l,angle_l = zip(*hessen_lines)
            # print
            max_s,min_s = max(slope_l),min(slope_l)
            max_a,min_a = max(angle_l),min(angle_l)

            # normalize values
            hessen_lines = [((max_s-s)/(max_s-min_s),(max_a-a)/(max_a-min_a)) for s,a in hessen_lines]
            # print hessen_lines

            tree = spatial.KDTree(hessen_lines)

            to_merge = []
            will_be_merged = set()

            for l_index in range(len(hessen_lines)-1,-1,-1):
                for l2_index in tree.query_ball_point(hessen_lines[l_index],0.15):
                    if l2_index > l_index:
                        t_lines = clusters[l_index][0][:]
                        t_lines.extend(clusters[l2_index][0])

                        aligned_text = self.__get_aggregation_lines__(t_lines)
                        accuracy = self.__agreement__(aligned_text)
                        if min(accuracy) >= 0.5:
                            will_be_merged.add(l_index)
                            will_be_merged.add(l2_index)

                            # make sure that there are not any overlapping users
                            users_1 = zip(*clusters[l_index][1])[1]
                            users_2 = zip(*clusters[l2_index][1])[1]

                            if [u for u in users_1 if u in users_2] != []:
                                continue

                            # is merge "relevant" to any other?
                            relevant = False
                            for m_index,m in enumerate(to_merge):
                                if (l_index in m) or (l2_index in m):
                                    relevant = True
                                    m.add(l_index)
                                    m.add(l2_index)
                                    break

                            if not relevant:
                                to_merge.append(set([l_index,l2_index]))

            # might be a better way to do this but will mulitple popping from list, safer
            # to work with a copy
            new_clusters = []

            for cluster_index in range(len(clusters)):
                if cluster_index not in will_be_merged:
                    new_clusters.append(clusters[cluster_index])
            for merged_clusters in to_merge:
                t_cluster = [[],[]]
                for cluster_index in merged_clusters:
                    t_cluster[0].extend(clusters[cluster_index][0])
                    t_cluster[1].extend(clusters[cluster_index][1])
                new_clusters.append(t_cluster[:])

            # print clusters
            clusters = new_clusters

        # and now, finally, the actual text clustering
        cluster_centers = []
        cluster_pts = []
        cluster_users = []

        cluster_members = []


        agreement = []
        for lines,pts_and_users in clusters:
            pts,users = zip(*pts_and_users)
            x1_values,x2_values,y1_values,y2_values = zip(*pts)

            # todo - handle when some of the coordinate values are not numbers -
            # this corresponds to when there are multiple text segments from the same user
            x1 = np.median(x1_values)
            x2 = np.median(x2_values)
            y1 = np.median(y1_values)
            y2 = np.median(y2_values)

            aligned_text = self.__get_aggregation_lines__(lines)
            aggregate_text = ""
            character_agreement = []
            for char_index in range(len(aligned_text[0])):
                char_set = set(text[char_index] for text in aligned_text)
                # get the percentage of votes for each character at this position
                char_vote = {c:sum([1 for text in aligned_text if text[char_index] == c])/float(len(aligned_text)) for c in char_set}
                most_likely_char,vote_percentage = max(char_vote.items(),key=lambda x:x[1])

                character_agreement.append(vote_percentage)

                if vote_percentage > 0.75:
                    aggregate_text += most_likely_char
                else:
                    aggregate_text += "Z"


            cluster_centers.append((x1,x2,y1,y2,aggregate_text))
            cluster_pts.append(zip(pts,lines))
            cluster_users.append(users)
            agreement.append(np.mean(character_agreement))

            # todo to remove all special characters from aligned_text
            cluster_members.append(aligned_text)
            # self.line_agreement.append((np.mean(character_agreement),len(users)))

        results = []
        for center,pts,users,lines,a in zip(cluster_centers,cluster_pts,cluster_users,cluster_members,agreement):
            results.append({"center":center,"cluster members":lines,"tools":[],"num users":len(users),"agreement":a})

        # return (cluster_centers,cluster_pts,cluster_users),0
        return results,0

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


def relevant_text_params(marking,image_dimensions):
    if ("startPoint" not in marking) or ("endPoint" not in marking):
        raise InvalidMarking(marking)
    x1 = marking["startPoint"]["x"]
    y1 = marking["startPoint"]["y"]
    x2 = marking["endPoint"]["x"]
    y2 = marking["endPoint"]["y"]

    if min(x1,x2,y1,y2) < 0:
        raise InvalidMarking(marking)

    if "text" not in marking:
        raise InvalidMarking(marking)

    text = marking["text"]

    if x1 <= x2:
        return x1,x2,y1,y2,text
    else:
        return x2,x2,y2,y1,text


def text_line_reduction(line_segments):
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


class SubjectRetirement(Classification):
    def __init__(self,param_dict):
        Classification.__init__(self)
        assert isinstance(param_dict,dict)

        # to retire subjects, we need a connection to the host api, which hopefully is provided
        self.host_api = None
        self.project_id = None
        self.token = None
        self.workflow_id = None
        for key,value in param_dict.items():
            if key == "host":
                self.host_api = value
            elif key == "project_id":
                self.project_id = value
            elif key == "token":
                self.token = value
            elif key == "workflow_id":
                self.workflow_id = value

        assert (self.host_api is not None) and (self.project_id is not None) and (self.token is not None) and (self.workflow_id is not None)

    def __task_aggregation__(self,classifications,gold_standard={}):
        to_retire = []
        for subject_id in classifications:
            users,everything_transcribed = zip(*classifications[subject_id])
            # count how many people have said everything is transcribed
            count = sum([1. for e in everything_transcribed if e == True])
            # and perent
            percent = sum([1. for e in everything_transcribed if e == True]) / float(len(everything_transcribed))
            if (count >= 3) and (percent >= 0.6):
                to_retire.append(subject_id)

        headers = {"Accept":"application/vnd.api+json; version=1","Content-Type": "application/json", "Authorization":"Bearer "+self.token}
        params = {"retired_subjects":to_retire}
        r = requests.post("https://panoptes.zooniverse.org/api/workflows/"+str(self.workflow_id)+"/links/retired_subjects",headers=headers,json=params)
        
        return []


class Tate(AggregationAPI):
    def __init__(self):
        AggregationAPI.__init__(self,245)#"tate",environment="staging")

        self.marking_params_per_shape = dict()

        self.marking_params_per_shape["text"] = relevant_text_params

        reduction_algs = {"text":text_line_reduction}
        self.__set_clustering_algs__({"text":TextCluster},reduction_algs)
        self.__set_classification_alg__(SubjectRetirement,{"host":self.host_api,"project_id":self.project_id,"token":self.token,"workflow_id":121})

        self.ignore_versions = True
        self.instructions[683] = {}

    def __readin_tasks__(self,workflow_id):
        marking_tasks = {"T2":["text"]}
        classification_tasks = {"T3" : True}

        return classification_tasks,marking_tasks

    def __plot_individual_points__(self,subject_id,task_id,shape):
        print self.cluster_algs["text"]

    def __get_subjects_to_aggregate__(self,workflow_id,with_expert_classifications=None):
        """
        override the retired subjects function to get only subjects which have been transcribed since we last ran
        the code
        :param workflow_id:
        :param with_expert_classifications:
        :return:
        """
        recently_classified_subjects = set()
        select = "SELECT subject_id,created_at from classifications where project_id="+str(self.project_id)

        for r in self.cassandra_session.execute(select):
            subject_id = r.subject_id
            if r.created_at >= self.old_time:
                recently_classified_subjects.add(subject_id)
        # assert False
        return list(recently_classified_subjects)

    def __prune__(self,aggregations):
        assert isinstance(aggregations,dict)
        for task_id in aggregations:
            if task_id == "param":
                continue

            if isinstance(aggregations[task_id],dict):
                for cluster_type in aggregations[task_id]:
                    if cluster_type == "param":
                        continue

                    del aggregations[task_id][cluster_type]["all_users"]
                    for cluster_index in aggregations[task_id][cluster_type]:
                        if cluster_index == "param":
                            continue

        return aggregations

if __name__ == "__main__":
    with Tate() as project:
        # print project.cluster_algs["text"].line_agreement
        project.__migrate__()
        project.__aggregate__(workflows=[121])
        # agreement_with_3 = [a for (a,l) in project.cluster_algs["text"].line_agreement if l >= 3]
        # print len(agreement_with_3)
        # print np.mean(agreement_with_3), np.median(agreement_with_3)
