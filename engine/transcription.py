#!/usr/bin/env python
from aggregation_api import AggregationAPI
import helper_functions
from classification import Classification
import clustering
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
import abc
import re
import random
import unicodedata
import os
import requests
import rollbar
import json
import sys
import yaml
from blob_clustering import BlobClustering
import boto3
import pickle
import getopt
from dateutil import parser
import json_transcription
import botocore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import matplotlib.cbook as cbook
import helper_functions

__author__ = 'greg'

folger_replacements = {}
# with open("/home/ggdhines/alpha_folger","rb") as alpha_folger:
#     with open("/home/ggdhines/beta_folger","rb") as beta_folger:
#         for l1,l2 in zip(alpha_folger.readlines(),beta_folger.readlines()):
#             l1 = l1.strip()
#             l2 = l2.strip()
#             folger_replacements[l1] = l2

# from https://gist.github.com/richarvey/637cd595362760858496
def get_signed_url(time, bucket, obj):
    s3 = boto3.resource('s3')

    url = s3.generate_url(
        time,
        'GET',
        bucket,
        obj,
        response_headers={
          'response-content-type': 'application/octet-stream'
        }
    )
    return url

# def folger_alpha_tags(text):
#     assert isinstance(text,str)
#     if "with" in text:
#         print text
#         a = True
#     else:
#         a = False
#     for l1,l2 in folger_replacements.items():
#         text = text.replace(l1,l2)
#
#     if a:
#         print text
#         print
#     return text

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

# def Levenshtein(a,b):
#     """
#     Calculates the Levenshtein distance between a and b
#     :param a:
#     :param b:
#     :return:
#     """
#     assert isinstance(a,str)
#     assert isinstance(b,str)
#
#     n, m = len(a), len(b)
#     if n > m:
#         # Make sure n <= m, to use O(min(n,m)) space
#         a,b = b,a
#         n,m = m,n
#
#     current = range(n+1)
#     for i in range(1,m+1):
#         previous, current = current, [i]+[0]*n
#         for j in range(1,n+1):
#             add, delete = previous[j]+1, current[j-1]+1
#             change = previous[j-1]
#             if a[j-1] != b[i-1]:
#                 change = change + 1
#             current[j] = min(add, delete, change)
#
#     return current[n]

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

#
# def set_depth(node,depth=0):
#     assert isinstance(node,AbstractNode)
#     node.__set_depth__(depth)
#
#     if node.rchild is not None:
#         set_depth(node.rchild,depth+1)
#     if node.lchild is not None:
#         set_depth(node.lchild,depth+1)
#
#
# def lowest_common_ancestor(node1,node2):
#     assert isinstance(node1,LeafNode)
#     assert isinstance(node2,LeafNode)
#
#     depth1 = node1.depth
#     depth2 = node2.depth
#
#     # make sure that the first node is the "shallower" node
#     if depth1 > depth2:
#         temp = node2
#         node2 = node1
#         node1 = temp
#
#         depth1 = node1.depth
#         depth2 = node2.depth
#     while depth2 > depth1:
#         node2 = node2.parent
#         depth2 = node2.depth
#
#     while node1 != node2:
#         node1 = node1.parent
#         node2 = node2.parent
#
#     return node1.height
#
#
# def create_clusters(ordering,maxima):
#     if maxima == []:
#         return [ordering,]
#
#     next_maxima = max(maxima,key=lambda x:x[1])
#
#     split = next_maxima[0]
#     left_split = ordering[:split]
#     right_split = ordering[split:]
#
#     maxima_index = maxima.index(next_maxima)
#     left_maximia = maxima[:maxima_index]
#     right_maximia = maxima[maxima_index+1:]
#     # need to adjust the indices for the right hand values
#     right_maximia = [(i-split,j) for (i,j) in right_maximia]
#
#     retval = create_clusters(left_split,left_maximia)
#     retval.extend(create_clusters(right_split,right_maximia))
#
#     return retval


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
    def __init__(self,shape,project,param_dict):
        clustering.Cluster.__init__(self,shape,project,param_dict)
        self.line_agreement = []

        self.tags = dict()
        tag_counter = 149

        if "tags" in param_dict:
            with open(param_dict["tags"],"rb") as f:
                for l in f.readlines():
                    tag = l[:-1]
                    assert isinstance(tag,str)
                    self.tags[tag_counter] = tag
                    tag_counter += 1

        self.erroneous_tags = dict()

        # stats to report back
        self.stats["capitalized"] = 0
        self.stats["double_spaces"] = 0
        self.stats["errors"] = 0
        self.stats["characters"] = 0

        self.stats["retired lines"] = 0



    def __line_alignment__(self,lines):
        """
        align.py the text by using MAFFT
        :param lines:
        :return:
        """

        if len(lines) == 1:
            return lines

        # todo - try to remember why I give each output file an id
        id_ = str(random.uniform(0,1))

        # with open(base_directory+"/Databases/transcribe"+id_+".fasta","wb") as f:
        with open("/tmp/transcribe"+id_+".fasta","wb") as f:
            for line in lines:
                if isinstance(line,tuple):
                    # we have a list of text segments which we should join together
                    line = "".join(line)

                # line = unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                # assert isinstance(line,str)

                # for i in range(max_length-len(line)):
                #     fasta_line += "-"

                try:
                    f.write(">\n"+line+"\n")
                except UnicodeEncodeError:
                    print line
                    print unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                    raise

        # todo - play around with gap penalty --op 0.5
        t = "mafft --op 0.2 --text /tmp/transcribe"+id_+".fasta> /tmp/transcribe"+id_+".out 2> /dev/null"
        os.system(t)

        aligned_text = []
        with open("/tmp/transcribe"+id_+".out","rb") as f:
            cumulative_line = ""
            for line in f.readlines():
                if (line == ">\n"):
                    if (cumulative_line != ""):
                        aligned_text.append(cumulative_line)
                        cumulative_line = ""
                else:
                    cumulative_line += line[:-1]

            if cumulative_line == "":
                print lines
                assert False
            aligned_text.append(cumulative_line)

        os.remove("/tmp/transcribe"+id_+".fasta")
        os.remove("/tmp/transcribe"+id_+".out")

        return aligned_text

    def __accuracy__(self,s):
        assert isinstance(s,str)
        assert len(s) > 0
        return sum([1 for c in s if c != "-"])/float(len(s))

    # # todo - is this function really necessary?
    # def __agreement__(self,text):
    #     """
    #     calculate the % of characters in each line of text where all characters at this position (over all lines)
    #     are in agreement. I ignore any starting or trailing "-" (spaces inserted for alignment)
    #     :param text:
    #     :return:
    #     """
    #     assert isinstance(text,list)
    #     assert len(text) > 1
    #     assert isinstance(text[0],str)
    #     assert min([len(t) for t in text]) == max([len(t) for t in text])
    #
    #     retval = []
    #
    #     for t in text:
    #         leftmost_char = -1
    #         rightmost_char = -1
    #         for i,c in enumerate(t):
    #             if c != "-":
    #                 leftmost_char = i
    #                 break
    #         for i,c in reversed(list(enumerate(t))):
    #             if c != "-":
    #                 rightmost_char = i
    #                 break
    #
    #         agreement = 0
    #
    #         for i in range(leftmost_char,rightmost_char+1):
    #             c = [t2[i].lower() for t2 in text]
    #             if min(c) == max(c):
    #                 assert c[0] != "-"
    #                 agreement += 1
    #
    #         retval.append(agreement/float(rightmost_char-leftmost_char+1))
    #     return retval

    # todo - can probably remove this function but double check
    # def __complete_agreement__(self,text):
    #     assert isinstance(text,list)
    #     assert len(text) > 1
    #     assert isinstance(text[0],str)
    #     assert min([len(t) for t in text]) == max([len(t) for t in text])
    #
    #     agreement = 0
    #     for i in range(len(text[0])):
    #
    #         c = [t[i].lower() for t in text if t[i] != "-"]
    #         if min(c) == max(c):
    #             assert c[0] != "-"
    #             agreement += 1
    #
    #     return agreement/float(len(text[0]))

    def __set_tags__(self,text):
        # convert to ascii
        text = text.encode('ascii','ignore')

        # good place to check if there is a newline character in the transcription
        # if so, someone tried to transcribe multiple lines at once - this is no longer allowed
        # but there are some legacy transcriptions with \n - such transcriptions are simply ignored
        if "\n" in text:
            return ""

        # the order of the keys matters - we need them to constant across all uses cases
        # we could sort .items() but that would be a rather large statement
        # replace each tag with a single non-standard ascii character (given by chr(num) for some number)
        text = text.strip()

        for key in sorted(self.tags.keys()):
            tag = self.tags[key]
            text = re.sub(tag,chr(key),text)

        # get rid of some other random tags and commands that shouldn't be included at all
        # todo - generalize
        text = re.sub("<br>","",text)
        text = re.sub("<font size=\"1\">","",text)
        text = re.sub("</font>","",text)
        text = re.sub("&nbsp","",text)
        text = re.sub("&amp","&",text)
        text = re.sub("\?\?\?","",text)

        return text

    def __set_special_characters__(self,text):
        """
        use upper case letters to represent special characters which MAFFT cannot deal with
        return a string where upper case letters all represent special characters
        "A" is used to represent all tags
        """

        # lower text is what we will give to MAFFT - it can contain upper case letters but those will
        # all represent something special, e.g. a tag
        lower_text = text.lower()

        # for lower_text, every tag will be represented by "A" - MAFFT cannot handle characters with
        # a value of greater than 127. To actually determine which tag we are talking about
        # will have to refer to text
        # strings are immutable in Python so we have to rebuild from scratch
        mafft_safe_text = ""
        for i,c in enumerate(lower_text):
            if ord(c) > 127:
                mafft_safe_text += "A"

            else:
                mafft_safe_text += c

        # take care of other characters which MAFFT cannot handle
        # note that text contains the original characters
        mafft_safe_text = re.sub(" ","I",mafft_safe_text)
        mafft_safe_text = re.sub("=","J",mafft_safe_text)
        mafft_safe_text = re.sub("\*","K",mafft_safe_text)
        mafft_safe_text = re.sub("\(","L",mafft_safe_text)
        mafft_safe_text = re.sub("\)","M",mafft_safe_text)
        mafft_safe_text = re.sub("<","N",mafft_safe_text)
        mafft_safe_text = re.sub(">","O",mafft_safe_text)
        mafft_safe_text = re.sub("-","P",mafft_safe_text)
        mafft_safe_text = re.sub("\'","Q",mafft_safe_text)

        return mafft_safe_text

    def __merge_aligned_text__(self,aligned_text):
        """
        once we have aligned the text using MAFFT, use this function to actually decide on an aggregate
        result - will also return the % of agreement
        and the percentage of how many characters for each transcription agree with the agree
        handles special tags just fine - and assumes that we have dealt with capitalization already
        """
        aggregate_text = ""
        num_agreed = 0

        # self.stats["line_length"].append(len(aligned_text[0]))

        vote_history = []

        for char_index in range(len(aligned_text[0])):
            # get all the possible characters
            # todo - we can reduce this down to having to loop over each character once
            # todo - handle case (lower case vs. upper case) better
            char_set = set(text[char_index] for text in aligned_text)
            # get the percentage of votes for each character at this position
            char_vote = {c:sum([1 for text in aligned_text if text[char_index] == c])/float(len(aligned_text)) for c in char_set}
            vote_history.append(char_vote)
            # get the most common character (also the most likely to be the correct one) and the percentage of users
            # who "voted" for it
            most_likely_char,vote_percentage = max(char_vote.items(),key=lambda x:x[1])

            # note that the most likely char can be 201 - which corresponds to inserted gaps
            # this corresponds to the case where one or more users gave some text but the rest of
            # of the users say there wasn't anything there
            if vote_percentage >= 0.65:
                aggregate_text += most_likely_char

            # check for special cases with double spaces or only differences about capitalization
            elif len(char_vote) == 2:
                sorted_keys = [c for c in sorted(char_vote.keys())]
                # 24 means a gap
                if (ord(sorted_keys[0]) == 24) and (sorted_keys[1] == " "):
                    # print most_likely_char,vote_percentage
                    raw_counts = {c:sum([1 for text in aligned_text if text[char_index] == c]) for c in char_set}
                    if raw_counts[chr(24)] >= 2:
                        # if at least two people said there was no space, we will assume that this space is actually
                        # a double space. So insert a gap - inserting a gap instead of just skipping it means
                        # that everything should stay the same length
                        aggregate_text += chr(24)
                        # self.stats["double_spaces"] += 1
                    else:
                        # this line is probably going to be counted as noise anyways, but just to be sure
                        aggregate_text += chr(27)
                        # self.stats["errors"] += 1
                # capitalization issues?
                elif sorted_keys[0].lower() == sorted_keys[1].lower():
                    aggregate_text += sorted_keys[0].upper()
                    # self.stats["capitalized"] += 1
                else:
                    aggregate_text += chr(27)
                    # self.stats["errors"] += 1
            else:
                # "Z" represents characters which we are not certain about
                aggregate_text += chr(27)
                # self.stats["errors"] += 1

        # what percentage of characters have we reached consensus on - i.e. we are fairly confident about?
        if len(aggregate_text) == 0:
            percent_consensus = -1
        else:
            # this is just equal to all the characters where are not equal to chr(27)
            percent_consensus = len([c for c in aggregate_text if ord(c) != 27])/float(len(aggregate_text))

        return aggregate_text,percent_consensus

    def __add_alignment_spaces__(self,aligned_text_list,capitalized_text):
        """
        take the text representation where we still have upper case and lower case letters
        plus special characters for tags (so definitely not the input for MAFFT) and add in whatever
        alignment characters are needed (say char(201)) so that the first text representations are all
        aligned
        fasta is the format the MAFFT reads in from - so non_fasta_text contains non-alpha-numeric ascii chars
        pts_and_users is used to match text in aligned text with non_fasta_text
        """

        aligned_nf_text_list = []
        for text,nf_text in zip(aligned_text_list,capitalized_text):
            aligned_nf_text = ""
            i = 0
            for c in text:
                if c == "-":
                    aligned_nf_text += chr(24)
                else:
                    aligned_nf_text += nf_text[i]
                    i += 1
            aligned_nf_text_list.append(aligned_nf_text)

        return aligned_nf_text_list

    # def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions,recusrive=False):
    #     unique_users = set(user_ids)
    #     if len(unique_users) <= 2:
    #         return [],0
    #
    #     clusters_by_indices  = self.__get_clusters_indices(reduced_markings)
    #
    #     X1,X2,Y1,Y2,transcriptions = zip(*markings)
    #     pts = zip(X1,X2,Y1,Y2)
    #     pca = PCA(n_components=2)
    #     pca.fit(pts)
    #     if sum(pca.explained_variance_ratio_) > 0.95:
    #         xy = pca.transform(pts)
    #         print pca.explained_variance_ratio_
    #
    #
    #
    #
    #         # print user_ids
    #
    #         colours = plt.cm.Spectral(np.linspace(0, 1, len(unique_users)))
    #
    #         for id_,c in zip(unique_users,colours):
    #             marked_by_user = [i for i,user_id in enumerate(user_ids) if user_id == id_]
    #
    #             plt.plot(xy[marked_by_user, 0], xy[marked_by_user, 1],"o",color=c)
    #         plt.show()
    #
    #         Z = hierarchy.linkage(xy, 'ward')
    #
    #         clusters_by_user_ids = [[id_,] for id_ in user_ids]
    #         for index1,index2,_,_ in Z:
    #             child1 = clusters_by_user_ids[int(index1)]
    #             child2 = clusters_by_user_ids[int(index2)]
    #
    #             if (child1 is None) and (child2 is None):
    #                 clusters_by_user_ids.append(None)
    #             elif child1 is None:
    #                 clusters_by_user_ids.append(None)
    #             elif child2 is None:
    #                 clusters_by_user_ids.append(None)
    #             else:
    #                 overlap = [i for i in child1 if i in child2]
    #
    #                 if overlap != []:
    #                     clusters_by_user_ids.append(None)
    #                 else:
    #                     combined_users = child1[:]
    #                     combined_users.extend(child2)
    #                     clusters_by_user_ids.append(combined_users)
    #
    #         print clusters_by_user_ids
    #
    #
    #
    #
    #         plt.figure()
    #         dn = hierarchy.dendrogram(Z)
    #         # fig, ax1 = plt.subplots(1, 1)
    #         dn1 = hierarchy.dendrogram(Z)#, ax=ax1)
    #         plt.show()
    #
    #         leven_distance_matrix = np.zeros((len(transcriptions),len(transcriptions)))
    #         for i1,t1 in enumerate(transcriptions):
    #             for i2,t2 in list(enumerate(transcriptions))[i1+1:]:
    #                 leven = Levenshtein(t1,t2)
    #                 leven_distance_matrix[i1,i2] = leven
    #                 leven_distance_matrix[i2,i1] = leven
    #
    #         Z = hierarchy.linkage(leven_distance_matrix,'ward')
    #         print Z
    #
    #     clusters = []
    #     for list_of_indices in clusters_by_indices:
    #         c = self.__create_cluster__(markings,list_of_indices)
    #         if c["num users"] > 2:
    #
    #
    #             # print
    #             # assert False
    #             clusters.append(c)
    #
    #
    #
    #     return clusters,0

    def __reset_tags__(self,text):
        """
        with text, we will have tags represented by a single character (with ord() > 128 to indicate
        that something is special) Convert these back to the full text representation
        also take care of folger specific stuff right
        :param text:
        :return:
        """
        assert isinstance(text,str)

        # reverse_map = {v: k for k, v in self.tags.items()}
        # also go with something different for "not sure"
        # this matter when the function is called on the aggregate text
        # reverse_map[200] = chr(27)
        # and for gaps inserted by MAFFT
        # reverse_map[201] = chr(24)

        ret_text = ""

        for c in text:
            if ord(c) > 128:
                ret_text += self.tags[ord(c)]
            else:
                ret_text += c

        assert isinstance(text,str)
        return ret_text


    # def __normalize_lines__(self,intercepts,slopes):
    #     """
    #     normalize the lines so that the intercepts and slopes are all between 0 and 1
    #     makes cluster better
    #     also returns a dictionary which allows us to "unnormalize" lines so that we refer to the original values
    #     """
    #     mean_intercept = np.mean(intercepts)
    #     std_intercept = np.std(intercepts)
    #
    #     normalized_intercepts = [(i-mean_intercept)/std_intercept for i in intercepts]
    #
    #     mean_slopes = np.mean(slopes)
    #     std_slopes = np.std(slopes)
    #
    #     normalized_slopes = [(s-mean_slopes)/std_slopes for s in slopes]
    #
    #     return normalized_intercepts,normalized_slopes

    # def __get_clusters_indices(self,reduced_markings):
    #     """
    #     get a list of the clusters - where each cluster gives just the indices
    #     of the transcriptions in that cluster - so no text aggregation/alignmennt is actually happening here
    #     actually I lied - there is some text alignment going on here to determine whether or not to add
    #     a transcription to a cluster - this will need to "rehappen" else where to do the actual aggregation
    #     which is slight redundant but I think will make things a lot easier for now
    #     """
    #     # start by splitting markings into lines and text and then the lines into slopes and intercepts
    #     intercepts,slopes,text = zip(*reduced_markings)
    #
    #     # deal with special characters in the text and "recombine" the markings
    #     # text has capital letters used only for special characters/tags
    #     # while capitalized_text has the original capitalization which is useful for the final aggregate result
    #     text,capitalized_text = zip(*[self.__set_special_characters__(t) for t in text])
    #
    #     # normalize the the slopes and intercepts
    #     normalized_intercepts,normalized_slopes = self.__normalize_lines__(intercepts,slopes)
    #     pts_list = zip(normalized_intercepts,normalized_slopes)
    #     # pts_list = zip(intercepts,slopes)
    #
    #     # see http://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
    #     labels = range(len(pts_list))
    #     variables = ["X","Y"]
    #     # X = np.random.random_sample([5,3])*10
    #     df = pd.DataFrame(list(pts_list),columns=variables, index=labels)
    #
    #     assigned_to_cluster = []
    #
    #     # variables = ["X"]
    #     # # X = np.random.random_sample([5,3])*10
    #     # df = pd.DataFrame(list(normalized_intercepts),columns=variables, index=labels)
    #
    #     # row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
    #     dist_matrix = squareform(pdist(df, metric='euclidean'))
    #
    #     clusters_by_indices = []
    #
    #     for transcription_index in range(len(text)):
    #         if transcription_index in assigned_to_cluster:
    #             continue
    #
    #         if text[transcription_index] == "":
    #             continue
    #
    #         assigned_to_cluster.append(transcription_index)
    #         transcriptions = [text[transcription_index]]
    #
    #         indices = [transcription_index]
    #
    #         distances = dist_matrix[transcription_index]
    #         distances = zip(range(len(distances)),distances)
    #
    #         distances.sort(key = lambda x:x[1])
    #
    #         # [1:] since the first element will be itself
    #         skips = 0
    #         allowable_skips = 5
    #         # only look at the next 20 closest items
    #         for ii,d in distances[1:20]:
    #             if ii not in assigned_to_cluster:
    #                 if text[ii] == "":
    #                     continue
    #
    #                 # create a new temp. set of transcriptions by adding in this next transcription
    #                 # temp_transcriptions = transcriptions[:]
    #                 # temp_transcriptions.append(text[ii])
    #
    #                 # check to see what the minimum accuracy is
    #                 # if high, go with this new set of transcriptions
    #                 # otherwise - allow one skip
    #                 # aligned_transcriptions = self.__line_alignment__(temp_transcriptions)
    #                 # accuracy = self.__agreement__(aligned_transcriptions)
    #
    #                 # print accuracy
    #                 max_leven = 0
    #
    #                 for trans in transcriptions:
    #                     leven = Levenshtein(trans,text[ii])
    #                     normalized_leven1 = leven/float(len(trans))
    #                     normalized_leven2 = leven/float(len(text[ii]))
    #
    #                     max_leven = max(max_leven,max(normalized_leven1,normalized_leven2))
    #
    #                     if max_leven >= 0.6:
    #                         break
    #
    #                 if max_leven < 0.6:
    #                     transcriptions.append(text[ii])
    #                     indices.append(ii)
    #                     assigned_to_cluster.append(ii)
    #                 else:
    #                     skips += 1
    #                     if skips == allowable_skips:
    #                         clusters_by_indices.append(indices)
    #                         break
    #         if skips < allowable_skips:
    #             clusters_by_indices.append(indices)
    #     # assert False
    #     return clusters_by_indices

def merge_clusters_with_distinct_users(cluster1,cluster2,dist):
    """
    given two clusters - where each element in a tuple is a user id
    - we might have more such as the corresponding marking index which doesn't matter
    merge these clusters if and only if there are no common users between the clusters
    :param cluster1:
    :param cluster2:
    :param dist:
    :return:
    """
    cluster1_users = zip(*cluster1)[0]
    cluster2_users = zip(*cluster2)[0]
    overlap = [id_ for id_ in cluster1_users if id_ in cluster2_users]

    return overlap == []


def merge_close_clusters(threshold):
    def f(cluster1,cluster2,dist):
        return dist <= threshold

    return f


def agglomerative_merge(cluster_tree,Z,merge_func):
    for index1,index2,dist,_ in Z:
        index1 = int(index1)
        index2 = int(index2)

        child1 = cluster_tree[index1]
        child2 = cluster_tree[index2]

        if (child1 is None) or (child2 is None):
            cluster_tree.append(None)
        else:

            if merge_func(child1,child2,dist):
                combined_users = child1[:]
                combined_users.extend(child2)
                cluster_tree.append(combined_users)

                cluster_tree[index1] = None
                cluster_tree[index2] = None
            else:
                cluster_tree.append(None)

    return cluster_tree

class Annotate(TextCluster):
    def __init__(self,shape,project,param_dict):
        TextCluster.__init__(self,shape,project,param_dict)

    def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions,subject_id):
        unique_users = set(user_ids)
        if len(unique_users) <= 2:
            return [],0

        # to create the clusters, we need two things - the user id corresponding to each
        # transcription. And the index of the transcription (out of the list of all transcriptions)
        # Actually only need the user id to create the clusters, but the transcription index is needed
        # to do anything after that
        cluster_tree = []
        # pts is for the clustering
        pts = []
        # also check that we still have least 3 unique users after removing any problem transcriptions
        unique_users = set()
        for i,id_ in enumerate(user_ids):
            text = markings[i][-1]
            # check to see if the text is empty
            # - either actually an empty transcription - not sure how likely that is, but better safe than sorry
            # - contains \n, in which case __set_special_characters__ returns ""
            # - is just some font tag stuff which is just ignored
            if self.__set_tags__(text) != "":
                cluster_tree.append([(id_,i),])
                pts.append(markings[i][:-1])
                unique_users.add(id_)

        if len(unique_users) <= 2:
            return [],0

        # todo - play around with n_components, if e_v_r is than 0.95, maybe increase to 3?
        pca = PCA(n_components=2)
        pca.fit(pts)
        print "explained variance is " + str(sum(pca.explained_variance_ratio_))
        xy = pca.transform(pts)

        # do agglomerative clustering - right now based on ward distance
        Z = hierarchy.linkage(xy, 'ward')

        cluster_tree = agglomerative_merge(cluster_tree,Z,merge_clusters_with_distinct_users)

        clusters_retval = []
        for cluster in cluster_tree:
            # if None - this node is either above or below a "cap" node
            if cluster is None:
                continue
            if len(cluster) <= 2:
                continue

            # extract the markings for this specific cluster
            markings_in_cluster = [markings[index] for (_,index) in cluster]

            # now create the cluster
            next_cluster,percentage_agreement = self.__create_cluster__(markings_in_cluster)

            aggregate_text = next_cluster["center"][-1]
            # gap characters shouldn't really count towards overall consensus
            # so if the aggregate text is just gap characters or "uncertain" characters
            # treat this a bad cluster
            non_gap_disagreement_characters = len([c for c in aggregate_text if ord(c) not in [24,27]])

            if (percentage_agreement < 0.5) or (non_gap_disagreement_characters == 0):
                next_clusters,found_clusters = self.__additional_clustering__(markings_in_cluster)

                # out of the mess of clusters, we found some "better" ones
                # they may still have low consensus - but we'll let the experts figure it out
                if next_clusters != []:
                    clusters_retval.extend(next_clusters)
                elif found_clusters is False:
                    # we know we have a problem cluster, so again, let the experts figure it out
                    clusters_retval.append(next_cluster)

                # image_file = self.project.__image_setup__(subject_id)[0]
                # image = plt.imread(image_file)
                # fig, ax1 = plt.subplots(1, 1)
                # ax1.imshow(image)
                #
                # for x1,x2,y1,y2,_ in markings_in_cluster:
                #     plt.plot([x1,x2],[y1,y2])
                #
                # plt.show()
            else:
                clusters_retval.append(next_cluster)

        print "lines retired: " + str(len(clusters_retval))
        return clusters_retval,0

    def __additional_clustering__(self,markings):
        """
        if we have a cluster with a low consensus - check what's happening
        are there transcriptions at a completely different angle? If so, put those
        into their own cluster. Any others will be treated like noise
        If no splitting happens - keep the problem cluster together - let the expert deal with it
        :param markings:
        :return:
        """
        # look for lines that have a really different angle
        # the ordering of the line coordinates is different for the hesse line reduction
        x1,x2,y1,y2,_ = zip(*markings)
        hesse_lines = helper_functions.hesse_line_reduction(zip(x1,y1,x2,y2))

        intercepts,angles = zip(*hesse_lines)
        angle_dist_matrix = np.zeros((len(markings),len(markings)))

        # the distance matrix is simply the distance between the line segments' angles
        for angle1 in range(len(markings)-1):
            for angle2 in range(angle1+1,len(markings)):
                distance = angles[angle1] - angles[angle2]
                angle_dist_matrix[angle1,angle2] = distance
                angle_dist_matrix[angle2,angle1] = distance

        Z = hierarchy.linkage(angle_dist_matrix, 'ward')
        cluster_tree = [[i,] for i in range(len(markings))]
        cluster_tree = agglomerative_merge(cluster_tree,Z,merge_close_clusters(0.25))

        # look for all clusters with at least two transcriptions in them
        # if we find any, apply levenshtein clustering on top of that
        second_pass_clusters = []

        for cluster in cluster_tree:
            # if None - this node is either above or below a "cap" node
            if cluster is None:
                continue
            # we just want at least 2 lines that are close together
            if len(cluster) >= 2:
                second_pass_clusters.append(cluster)

        if second_pass_clusters == []:
            # this is a problem set of transcriptions so create a cluster
            # and let the experts deal with it
            return [],False
        else:
            # only return aggregations for clusters with at least 3 transcriptions
            # if only 2 - don't report this as a problem cluster - hopefully with some more transcriptions
            # we will figure it out
            clusters_retval = []
            for cluster in second_pass_clusters:
                if len(cluster) >= 3:
                    markings_in_cluster = [markings[index] for index in cluster]

                    next_cluster = self.__create_cluster__(markings_in_cluster)
                    clusters_retval.append(next_cluster)

            return clusters_retval,True

    def __create_cluster__(self,markings_in_cluster):
        """
        given a list of markings in a single cluster, actually create that cluster
        i.e. aggregate the text, and create the cluster variable and add all the necessary info
        """

        x1_values,x2_values,y1_values,y2_values,transcriptions = zip(*markings_in_cluster)

        # first replace every tag (multi characters) with a single character token
        transcriptions = [self.__set_tags__(t) for t in transcriptions]

        # now deal with any characters with MAFFT has trouble with (i.e. replace " ", and non standard ascii characters)
        mafft_safe_trans = [self.__set_special_characters__(t) for t in transcriptions]

        # is everyone in agreement? - if so, don't bother even calling MAFFT
        # such external calls seem to be a problem on AWS so trying to minimize them
        # doing this now (as opposed to a few lines earlier) since I don't want capitalization differences
        # to affect things. The merge_aligned_text will still handle differences in capitalization but
        # we don't need to call MAFFT
        # note that if all the transcriptions are the same length we probably don't need to call MAFFT
        # but just to be safe, we will
        agreement = [mafft_safe_trans[i] == mafft_safe_trans[i+1] for i in range(len(mafft_safe_trans)-1)]

        # we don't have complete agreement (or near complete agreement) in the transcriptions
        if False in agreement:
            # call mafft to align these transcriptions
            aligned_transcriptions = self.__line_alignment__(mafft_safe_trans)
            # apply these alignments (insert gaps) to the original text with capitalization
            aligned_transcriptions = self.__add_alignment_spaces__(aligned_transcriptions,transcriptions)

            # in cases where there is disagreement, use voting to determine the most likely character
            # if there is strong disagreement, we'll mark those spots as unknown
            aggregate_text,percent_agreement= self.__merge_aligned_text__(aligned_transcriptions)
        else:
            # we know that we have near complete agreement (up to disagreement about capitalizations) so we know
            # that the text is aligned
            aggregate_text,percent_agreement = self.__merge_aligned_text__(transcriptions)

        # convert tags from tokens back into multicharacter strings (so people can read them)
        aggregate_text = self.__reset_tags__(aggregate_text)

        # find the average start and end
        x1 = np.median(x1_values)
        x2 = np.median(x2_values)
        y1 = np.median(y1_values)
        y2 = np.median(y2_values)

        # cluster is what we'll actually return
        cluster = dict()
        cluster["center"] = (x1,x2,y1,y2,aggregate_text)

        cluster["tools"] = []

        cluster["cluster members"] = markings_in_cluster

        cluster["num users"] = len(cluster["cluster members"])

        # collect some stats
        self.stats["retired lines"] += 1

        aggregate_text = cluster["center"][-1]

        errors = sum([1 for c in aggregate_text if ord(c) == 27])
        self.stats["errors"] += errors
        self.stats["characters"] += len(aggregate_text)

        return cluster,percent_agreement



class SubjectRetirement(Classification):
    def __init__(self,environment,param_dict):
        Classification.__init__(self,environment)
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

        self.num_retired = None
        self.non_blanks_retired = None

        self.to_retire = None

        assert (self.host_api is not None) and (self.project_id is not None) and (self.token is not None) and (self.workflow_id is not None)

    def __aggregate__(self,raw_classifications,workflow,aggregations):
        # start by looking for empty subjects

        self.to_retire = set()
        for subject_id in raw_classifications["T0"]:
            user_ids,is_subject_empty = zip(*raw_classifications["T0"][subject_id])
            if is_subject_empty != []:
                empty_count = sum([1 for i in is_subject_empty if i == True])
                if empty_count >= 3:
                    self.to_retire.add(subject_id)

        blank_retirement = len(self.to_retire)

        non_blanks = []

        # now look to see if everything has been transcribed
        for subject_id in raw_classifications["T3"]:
            user_ids,completely_transcribed = zip(*raw_classifications["T3"][subject_id])

            completely_count = sum([1 for i in completely_transcribed if i == True])
            if completely_count >= 3:
                self.to_retire.add(subject_id)
                non_blanks.append(subject_id)

            # # have at least 4/5 of the last 5 people said the subject has been completely transcribed?
            # recent_completely_transcribed = completely_transcribed[-5:]
            # if recent_completely_transcribed != []:
            #     complete_count = sum([1 for i in recent_completely_transcribed if i == True])/float(len(recent_completely_transcribed))
            #
            #     if (len(recent_completely_transcribed) == 5) and (complete_count >= 0.8):
            #         to_retire.add(subject_id)

        # don't retire if we are in the development environment
        if (self.to_retire != set()) and (self.environment != "development"):
            try:
                headers = {"Accept":"application/vnd.api+json; version=1","Content-Type": "application/json", "Authorization":"Bearer "+self.token}
                params = {"retired_subjects":list(self.to_retire)}
                # r = requests.post("https://panoptes.zooniverse.org/api/workflows/"+str(self.workflow_id)+"/links/retired_subjects",headers=headers,json=params)
                r = requests.post("https://panoptes.zooniverse.org/api/workflows/"+str(self.workflow_id)+"/links/retired_subjects",headers=headers,data=json.dumps(params))
                # rollbar.report_message("results from trying to retire subjects","info",extra_data=r.text)

            except TypeError as e:
                print e
                rollbar.report_exc_info()
        if self.environment == "development":
            print "we would have retired " + str(len(self.to_retire))
            print "with non-blanks " + str(len(self.to_retire)-blank_retirement)
            if not os.path.isfile("/home/ggdhines/"+str(self.project_id)+".retired"):
                pickle.dump(non_blanks,open("/home/ggdhines/"+str(self.project_id)+".retired","wb"))
            print str(len(self.to_retire)-blank_retirement)

        self.num_retired = len(self.to_retire)
        self.non_blanks_retired = len(self.to_retire)-blank_retirement

        return aggregations


class TranscriptionAPI(AggregationAPI):
    def __init__(self,project_id,environment,end_date=None):
        AggregationAPI.__init__(self,project_id,environment,end_date=end_date)



        self.rollbar_token = None

        # just to stop me from using transcription on other projects
        assert int(project_id) in [245,376]



    def __cluster__(self,used_shapes,raw_markings,image_dimensions):
        """
        run the clustering algorithm for a given workflow
        need to have already checked that the workflow requires clustering
        :param workflow_id:
        :return:
        """

        if raw_markings == {}:
            print "warning - empty set of images"
            # print subject_set
            return {}

        # start by clustering text
        print "clustering text"
        cluster_aggregation = self.text_algorithm.__aggregate__(raw_markings,image_dimensions)
        print "clustering images"
        image_aggregation = self.image_algorithm.__aggregate__(raw_markings,image_dimensions)

        cluster_aggregation = self.__merge_aggregations__(cluster_aggregation,image_aggregation)

        return cluster_aggregation

    def __setup__(self):
        AggregationAPI.__setup__(self)

        workflow_id = self.workflows.keys()[0]

        self.__set_classification_alg__(SubjectRetirement,{"host":self.host_api,"project_id":self.project_id,"token":self.token,"workflow_id":workflow_id})

        self.instructions[workflow_id] = {}

        self.marking_params_per_shape["text"] = helper_functions.relevant_text_params
        # the code to cluster lines together
        # self.default_clustering_algs["text"] = TextCluster
        # self.default_clustering_algs["image"] = BlobClustering

        # set up the text clusering algorithm
        additional_text_args = {"reduction":helper_functions.text_line_reduction}
        # load in the tag file if there is one

        api_details = yaml.load(open("/app/config/aggregation.yml","rb"))
        if "tags" in api_details[self.project_id]:
            additional_text_args["tags"] = api_details[self.project_id]["tags"]

        # we need the clustering algorithms to exist after they've been used (so we can later extract
        # some stats) - this is currently not the way its done with the aggregationAPI, so we'll do it
        # slightly differently

        if self.project_id == 245:
            self.text_algorithm = Annotate("text",self,additional_text_args)
        self.image_algorithm = BlobClustering("image",None,{})

        self.only_retired_subjects = False
        self.only_recent_subjects = True

    def __enter__(self):
        if self.environment != "development":
            panoptes_file = open("/app/config/aggregation.yml","rb")
            api_details = yaml.load(panoptes_file)
            self.rollbar_token = api_details[self.environment]["rollbar"]
            rollbar.init(self.rollbar_token,self.environment)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if (self.rollbar_token is not None) and (self.environment != "development") and (exc_type is not None):
        #     rollbar.report_exc_info()

            # return True
        pass

    def __cluster_output_with_colour__(self,workflow_id,ax,subject_id):
        """
        use colour to show where characters match and don't match between different transcriptions of
        the same text
        :param subject_id:
        :return:
        """
        selection_stmt = "SELECT aggregation FROM aggregations WHERE workflow_id = " + str(workflow_id) + " AND subject_id = " + str(subject_id)
        cursor = self.postgres_session.cursor()
        cursor.execute(selection_stmt)

        aggregated_text = cursor.fetchone()[0]["T2"]["text clusters"].values()
        assert isinstance(aggregated_text,list)
        # remove the list of all users
        aggregated_text = [a for a in aggregated_text if isinstance(a,dict)]

        # sort the text by y coordinates (should give the order in which the text is supposed to appear)
        aggregated_text.sort(key = lambda x:x["center"][2])

        for text in aggregated_text:
            ax.plot([text["center"][0],text["center"][1]],[text["center"][2],text["center"][3]],color="red")
            actual_text = text["center"][-1]
            atomic_text = self.cluster_algs["text"].__set_special_characters__(actual_text)[1]

            for c in atomic_text:
                if ord(c) == 27:
                    # no agreement was reached
                    print chr(8) + unicode(u"\u2224"),
                elif ord(c) == 28:
                    # the agreement was that nothing was here
                    # technically not a space but close enough
                    print chr(8) + " ",
                else:
                    print chr(8) + c,
            print

    def __readin_tasks__(self,workflow_id):
        if self.project_id == 245:
            # marking_tasks = {"T2":["image"]}
            marking_tasks = {"T2":["text","image"]}
            # todo - where is T1?
            classification_tasks = {"T0":True,"T3" : True}

            return classification_tasks,marking_tasks,{}
        elif self.project_id == 376:
            marking_tasks = {"T2":["text"]}
            classification_tasks = {"T0":True,"T3":True}

            print AggregationAPI.__readin_tasks__(self,workflow_id)

            return classification_tasks,marking_tasks,{}
        else:
            return AggregationAPI.__readin_tasks__(self,workflow_id)

    # def __get_subjects_to_aggregate__(self,workflow_id,with_expert_classifications=None):
    #     """
    #     override the retired subjects function to get only subjects which have been transcribed since we last ran
    #     the code
    #     :param workflow_id:
    #     :param with_expert_classifications:
    #     :return:
    #     """
    #     recently_classified_subjects = set()
    #     select = "SELECT subject_id,created_at from classifications where project_id="+str(self.project_id)
    #
    #     for r in self.cassandra_session.execute(select):
    #         subject_id = r.subject_id
    #         if r.created_at >= self.old_time:
    #             recently_classified_subjects.add(subject_id)
    #     # assert False
    #     return list(recently_classified_subjects)

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

    def __summarize__(self,tar_path=None):
        num_retired = self.classification_alg.num_retired
        non_blanks_retired = self.classification_alg.non_blanks_retired

        stats = self.text_algorithm.stats

        old_time_string = self.previous_runtime.strftime("%B %d %Y")
        new_time_string = end_date.strftime("%B %d %Y")

        accuracy =  1. - stats["errors"]/float(stats["characters"])

        subject = "Aggregation summary for " + str(old_time_string) + " to " + str(new_time_string)

        body = "This week we have retired " + str(num_retired) + " subjects, of which " + str(non_blanks_retired) + " where not blank."
        body += " A total of " + str(stats["retired lines"]) + " lines were retired. "
        body += " The accuracy of these lines was " + "{:2.1f}".format(accuracy*100) + "% - defined as the percentage of characters where at least 3/4's of the users were in agreement."

        print self.__panoptes_call__("projects/"+str(self.project_id)+"/aggregations_export?admin=true")
        assert False
        if tar_path is not None:
            bucket = "s3://zooniverse-static/panoptes-uploads.zooniverse.org/"+str(self.project_id)+"/"
            s3 = boto3.resource('s3')
            try:
                s3.meta.client.head_bucket(Bucket=bucket)
            except botocore.exceptions.ClientError as e:
                # If a client error is thrown, then check that it was a 404 error.
                # If it was a 404 error, then the bucket does not exist.
                error_code = int(e.response['Error']['Code'])
                if error_code == 404:
                    s3.create_bucket(Bucket=bucket,CreateBucketConfiguration={'LocationConstraint': 'us-west-1'})

            object = s3.Object('mybucket', 'hello.txt')
            object.put(Body=open('/tmp/hello.txt', 'rb'))

            url = get_signed_url(604800, bucket, object)





        body += "\n Greg Hines \n Zooniverse \n \n PS This email was automatically generated."

        client = boto3.client('ses')
        response = client.send_email(
            Source='greg@zooniverse.org',
            Destination={
                'ToAddresses': [
                    'greg@zooniverse.org','victoria@zooniverse.org','matt@zooniverse.org'
                ]#,
                # 'CcAddresses': [
                #     'string',
                # ],
                # 'BccAddresses': [
                #     'string',
                # ]
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'ascii'
                },
                'Body': {
                    'Text': {
                        'Data': body,
                        'Charset': 'ascii'
                    }
                }
            },
            ReplyToAddresses=[
                'greg@zooniverse.org',
            ],
            ReturnPath='greg@zooniverse.org'
        )

        print response

# retired_subjects = [649216, 649217, 649218, 649219, 649220, 649222, 649223, 649224, 649225, 671754, 649227, 649228, 649229, 649230, 667309, 662872, 649234, 653332, 649239, 653336, 671769, 673482, 653339, 649244, 649245, 649246, 649248, 653346, 653348, 653349, 665642, 653356, 649261, 649262, 649263, 671752, 649267, 671796, 649269, 649270, 649272, 649226, 653316, 649281, 669763, 669765, 669766, 669769, 653390, 649231, 669793, 650598, 649233, 662887, 667764, 662890, 649362, 649363, 649365, 649366, 649368, 663705, 649370, 649371, 649372, 649373, 663710, 663711, 671904, 671905, 649378, 665627, 663717, 671910, 671911, 671912, 663722, 663723, 649388, 663726, 663727, 673480, 663730, 671923, 649397, 671927, 649400, 671932, 667338, 671936, 653344, 671938, 673483, 671940, 665634, 649426, 669923, 669926, 669927, 669932, 669933, 669937, 649461, 669942, 649463, 671992, 669946, 669947, 649258, 649470, 669951, 649472, 669954, 669955, 669957, 649478, 669960, 669962, 671447, 649484, 669965, 649486, 669969, 649490, 669972, 669976, 649497, 669978, 669981, 669982, 669984, 669985, 669987, 649508, 669989, 649510, 649265, 669992, 672896, 669995, 649505, 649518, 649519, 649520, 649521, 649522, 649523, 649524, 649525, 649526, 649527, 649528, 649529, 649530, 649531, 649532, 649533, 649534, 649535, 672283, 649537, 649538, 649540, 649541, 649542, 649543, 649544, 649545, 672081, 672083, 672084, 672086, 672087, 649564, 649573, 672102, 662929, 672104, 672106, 649592, 649601, 663939, 663940, 672136, 672834, 671811, 673518, 653381, 649639, 649644, 670127, 649653, 649578, 662945, 649647, 662948, 649511, 649728, 649744, 649745, 649747, 664089, 664090, 664091, 672284, 672285, 672286, 662961, 672297, 649780, 672310, 672313, 649788, 649793, 649798, 649828, 650342, 672360, 649836, 672366, 650344, 649845, 649848, 672385, 649858, 672409, 649890, 649892, 662641, 649900, 672541, 649907, 672439, 672443, 672885, 649922, 649937, 649942, 649944, 664301, 664303, 649369, 650366, 664313, 603264, 649995, 664334, 603268, 664346, 664349, 603270, 672551, 672554, 603271, 653341, 672558, 672561, 672562, 603273, 672906, 672907, 672567, 664396, 664403, 649387, 653114, 668518, 668522, 672915, 664454, 664463, 664469, 664471, 664473, 664474, 664488, 649374, 672714, 668628, 671909, 650212, 670700, 650688, 671914, 671915, 666640, 649390, 671919, 672797, 649462, 672803, 672808, 672809, 672810, 672814, 672815, 672817, 661465, 652344, 652347, 652354, 653010, 672858, 650336, 672870, 662632, 650346, 672875, 650349, 650352, 650353, 650355, 650357, 650358, 650360, 650361, 650363, 603262, 603263, 650368, 650369, 672898, 650371, 650372, 650374, 650375, 650377, 650378, 603275, 672909, 603279, 603280, 603281, 603283, 662676, 603285, 603288, 649466, 650400, 603297, 603299, 603300, 650411, 650413, 667334, 672952, 649608, 672968, 672973, 672977, 672980, 672295, 653015, 670979, 670993, 670998, 671007, 671014, 671017, 671019, 671022, 671024, 671030, 671034, 671036, 671037, 662847, 671042, 662851, 671047, 671048, 671061, 662870, 671064, 662874, 650590, 671073, 650594, 650596, 671078, 671079, 650602, 671083, 662894, 662895, 671088, 671089, 662898, 671091, 662901, 671094, 671099, 671101, 671102, 662922, 671115, 662924, 662926, 671119, 667024, 671121, 671125, 671126, 671128, 671129, 662941, 662943, 662944, 671137, 671138, 662947, 671140, 662949, 662950, 662951, 662953, 662955, 662956, 662958, 662959, 671153, 671154, 671156, 662910, 671158, 671161, 671163, 649203, 671168, 671170, 671173, 671174, 673481, 671179, 650701, 671184, 671640, 671189, 650636, 671994, 671198, 653364, 649467, 671204, 671206, 671208, 671209, 671220, 671222, 649471, 671230, 671232, 649474, 671249, 671253, 649477, 671267, 671269, 650797, 649480, 671282, 650809, 649482, 671296, 671302, 671303, 653239, 671330, 653240, 671327, 667234, 667238, 667240, 667242, 667244, 667248, 667252, 667258, 649496, 649498, 663204, 667303, 667305, 667306, 667307, 667308, 663213, 667313, 667315, 667318, 671417, 667322, 667324, 667326, 649504, 667331, 667332, 667333, 663238, 667335, 667336, 667337, 663242, 663243, 653004, 663245, 673486, 671442, 653011, 667350, 667351, 667352, 671449, 671452, 671453, 663262, 649509, 673505, 663268, 663269, 663270, 649213, 663278, 673520, 649512, 663282, 663283, 650537, 649514, 663296, 649515, 671939, 671510, 653273, 663323, 663324, 663326, 650544, 663330, 663334, 663335, 663337, 671532, 653307, 673594, 653279, 673606, 671565, 673614, 671568, 673620, 665435, 653149, 653151, 653152, 653154, 671588, 662844, 671594, 673672, 653310, 653294, 671638, 665495, 665496, 671727, 673693, 649200, 670022, 653231, 651185, 653234, 651187, 671671, 651192, 651193, 653243, 651197, 671678, 649205, 671680, 651201, 653250, 651203, 665540, 653253, 671686, 653257, 665547, 653260, 671693, 665550, 665551, 653304, 653270, 665559, 663512, 665561, 665564, 671709, 661471, 665569, 653283, 653285, 671718, 653288, 671721, 671722, 653291, 649198, 649199, 653296, 649201, 649202, 653299, 671733, 649206, 649207, 649208, 649209, 649211, 649212, 653309, 649214, 649215]
# import random
# random.seed(0)
# retired_subjects = retired_subjects[:200]



if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"shi:e:d:",["summary","project_id=","environment=","end_date="])
    except getopt.GetoptError:
        print 'transcription.py -i <project_id> -e: <environment> -d: <end_date>'
        sys.exit(2)

    environment = "development"
    project_id = None
    end_date = None
    summary = False

    for opt, arg in opts:
        if opt in ["-i","--project_id"]:
            project_id = int(arg)
        elif opt in ["-e","--environment"]:
            environment = arg
        elif opt in ["-d","--end_date"]:
            end_date = parser.parse(arg)
        elif opt in ["-s","--summary"]:
            summary = True

    assert project_id is not None

    with TranscriptionAPI(project_id,environment,end_date) as project:
        project.__setup__()
        if environment == "production":
            project.__migrate__()
        # project.__migrate__()
        print "done migrating"
        # project.__aggregate__(subject_set = [671541,663067,664482,662859])
        project.__aggregate__()

        if summary:
            tar_path = json_transcription.json_dump(project)

            project.__summarize__(tar_path)

        # print aggregated_text