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
import cPickle as pickle

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

        self.tags = {}
        self.tags["\[deletion\]"] = chr(150)
        self.tags["\[/deletion\]"] = chr(151)
        self.tags["\[illegible\]"] = chr(152)
        self.tags["\[/illegible\]"] = chr(153)
        self.tags["\[insertion\]"] = chr(154)
        self.tags["\[/insertion\]"] = chr(155)
        self.tags["\[notenglish\]"] = chr(156)
        self.tags["\[/notenglish\]"] = chr(157)

    def __line_alignment__(self,lines):
        """
        align the text by using MAFFT
        :param lines:
        :return:
        """

        # todo - try to remember why I give each output file an id
        id_ = str(random.uniform(0,1))

        # with open(base_directory+"/Databases/transcribe"+id_+".fasta","wb") as f:
        with open("/tmp/transcribe"+id_+".fasta","wb") as f:
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

        t = "mafft --text /tmp/transcribe"+id_+".fasta> /tmp/transcribe"+id_+".out 2> /dev/null"
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

            assert cumulative_line != ""
            aligned_text.append(cumulative_line)

        os.remove("/tmp/transcribe"+id_+".fasta")
        os.remove("/tmp/transcribe"+id_+".out")

        return aligned_text

    def __accuracy__(self,s):
        assert isinstance(s,str)
        assert len(s) > 0
        return sum([1 for c in s if c != "-"])/float(len(s))

    # todo - is this function really necessary?
    def __agreement__(self,text):
        """
        calculate the % of characters in each line of text where all characters at this position (over all lines)
        are in agreement. I ignore any starting or trailing "-" (spaces inserted for alignment)
        :param text:
        :return:
        """
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

    def __set_special_characters__(self,text):
        """
        use upper case letters to represent special characters which MAFFT cannot deal with
        return a string where upper case letters all represent special characters
        "A" is used to represent all tags (all for an abitrary number of tags)
        so also return a dictionary which gives the index of each occurrence of "A" and what tag it originally referred to
        also return a string where all special characters are represented by capital letters BUT
        we also keep any capitalize found in the original line of text. Confusing but this will allow us
        to rebuild the capitalization at the very end
        """

        text = text.encode('ascii','ignore')

        # first we need to replace each tag with a one character representation
        for tag,chr_representation in self.tags.items():
            text = re.sub(tag,chr_representation,text)
            # lower_text = re.sub(tag,chr_representation,lower_text)

        # we can do this afterwards because non-alphabet characters are not affected by .lower
        lower_text = text.lower()

        # then record were each tag occurs
        # tag_indices = {}
        new_lower_text = ""
        for char_index in range(len(lower_text)):
            if ord(lower_text[char_index]) > 127:
                new_lower_text += "A"
                # tag_indices[char_index] = ord(lower_text[char_index])
            else:
                new_lower_text += lower_text[char_index]
        lower_text = new_lower_text

        # take care of other characters which MAFFT cannot handle
        # note that text contains the original characters
        lower_text = re.sub(" ","I",lower_text)
        lower_text = re.sub("=","J",lower_text)
        lower_text = re.sub("\*","K",lower_text)
        lower_text = re.sub("\(","L",lower_text)
        lower_text = re.sub("\)","M",lower_text)
        lower_text = re.sub("<","N",lower_text)
        lower_text = re.sub(">","O",lower_text)
        lower_text = re.sub("-","P",lower_text)
        lower_text = re.sub("\'","Q",lower_text)

        # text = re.sub(" ",chr(160),text)
        # text = re.sub("=",chr(161),text)
        # text = re.sub("\*",chr(162),text)
        # text = re.sub("\(",chr(163),text)
        # text = re.sub("\)",chr(164),text)
        # text = re.sub("<",chr(165),text)
        # text = re.sub(">",chr(166),text)
        # text = re.sub("-",chr(167),text)

        # todo - find a way to fix this - stupid postgres/json
        # text = re.sub(r'\'',chr(168),text)

        return lower_text,text

    def __reset_special_characters__(self,text):
        """
        python/postgres seem have a bit of a problem with non-standard ascii code
        not sure why - but the easiest way to deal with this seems to be reset each non-standard ascii code
        to the tag it corresponds to
        :param text:
        :return:
        """
        # use a 'reverse dictionary' to reset
        for (t,c) in self.tags.items():
            text = re.sub(c,t,text)

        # also go with something different for "not sure"
        text = re.sub(chr(200),chr(27),text)

        return text

    def __merge_aligned_text__(self,aligned_text):
        """
        once we have aligned the text using MAFFT, use this function to actually decide on an aggregate
        result - will also return the % of agreement
        and the percentage of how many characters for each transcription agree with the agree
        handles special tags just fine - and assumes that we have dealt with capitalization already
        """
        aggregate_text = ""
        num_agreed = 0

        # will keep track of the percentage of characters from each transcription which agree
        # with the aggregate
        agreement_per_user = [0 for i in aligned_text]

        for char_index in range(len(aligned_text[0])):
            # get all the possible characters
            # todo - we can reduce this down to having to loop over each character once
            # todo - handle case (lower case vs. upper case) better
            char_set = set(text[char_index] for text in aligned_text)
            # get the percentage of votes for each character at this position
            char_vote = {c:sum([1 for text in aligned_text if text[char_index] == c])/float(len(aligned_text)) for c in char_set}
            # get the most common character (also the most likely to be the correct one) and the percentage of users
            # who "voted" for it
            most_likely_char,vote_percentage = max(char_vote.items(),key=lambda x:x[1])

            # chr 201 - is for inserted spaces to align text. I don't think the most likely character is allowed
            # to be 201, but just to be safe
            if (vote_percentage > 0.75) and (most_likely_char != chr(201)):
                num_agreed += 1
                aggregate_text += most_likely_char

                for i in range(len(aligned_text)):
                    if aligned_text[i][char_index] == most_likely_char:
                        agreement_per_user[i] += 1
            else:
                # "Z" represents characters which we are not certain about
                aggregate_text += chr(200)

        # what percentage of characters have we reached consensus on - i.e. we are fairly confident about?
        percent_consensus = num_agreed/float(len(aggregate_text))

        # convert the agreement per user to a percentage
        agreement_per_user = [a/float(len(aggregate_text)) for a in agreement_per_user]

        return aggregate_text,percent_consensus,agreement_per_user

    def __add_alignment_spaces__(self,aligned_text_list,non_fasta_text_dict,pts_and_users):
        """
        take the text representation where we still have upper case and lower case letters
        plus special characters for tags (so definitely not the input for MAFFT) and add in whatever
        alignment characters are needed (say char(201)) so that the first text representations are all
        aligned
        fasta is the format the MAFFT reads in from - so non_fasta_text contains non-alpha-numeric ascii chars
        pts_and_users is used to match text in aligned text with non_fasta_text
        """
        # for text,nf_text in zip(aligned_text,non_fasta_text):
        #     print text,nf_text
        aligned_nf_text_list = []
        for text,user_ident in zip(aligned_text_list,pts_and_users):
            aligned_nf_text = ""
            nf_text = non_fasta_text_dict[user_ident]
            i = 0
            for c in text:
                if c == "-":
                    aligned_nf_text += chr(201)
                else:
                    aligned_nf_text += nf_text[i]
                    i += 1
            aligned_nf_text_list.append(aligned_nf_text)

        return aligned_nf_text_list

    def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions):
        # we want to first cluster first just on dist and theta - ignoring the text contents
        # dist_list,theta_list,text_list,raw_pts_list = zip(*markings)
        # mapped_markings = zip(dist_list,theta_list)

        # cluster just on points, not on text
        dist_l,theta_l,text_l = zip(*reduced_markings)
        reduced_markings_without_text = zip(dist_l,theta_l)
        ordering  = self.__preliminarily__clustering__(reduced_markings_without_text,user_ids)

        # use the below 2 to build up each cluster
        current_lines = {}
        current_pts = {}
        clusters = []

        non_fasta_text = {}

        for a,b in ordering:
            # a - line values - "intercept" and slope
            user_index = reduced_markings_without_text.index(a)
            user = user_ids[user_index]
            # extract the corresponding text and the raw (unmapped) point
            # text = text_list[user_index]
            # raw_pt = raw_pts_list[user_index]

            text = markings[user_index][-1]
            raw_pt = markings[user_index][:-1]

            # skip lines with new lines characters in them
            # Roger has set things up so that new line characters are no longer allowed
            # but we need to be careful with transcriptions submitted before that change
            if "\n" in text:
                print "multiline - skipping"
                continue

            # convert from unicode to ascii
            assert isinstance(text,unicode)

            # not sure if it is possible to have empty lines, but just in case
            if text == "":
                continue

            # handle all characters which MAFFT cannot handle and keep a record of where all
            # the tags are in the string
            # text_with_capitalization is used (at the end) to determine the correct capitalization
            # of character (since in the mean time capital letters are used for other stuff)
            text, nf_text = self.__set_special_characters__(text)

            # save these values for later use
            non_fasta_text[(raw_pt,user)] = nf_text

            # if we currently have an empty cluster, just add the line
            if current_lines == {}:
                current_lines[user] = text
                # adding the user id is slightly redundant but makes doing the actual clustering easier
                current_pts[user] = (raw_pt,user)
            else:
                # need to see if we want to merge the text with the existing cluster or start a new one
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
                    # todo - why am I sorting here? doesn't really seem necessary
                    # users_and_lines = sorted(current_lines.items(),key = lambda x:x[0])
                    # sorted_users,sorted_lines = zip(*users_and_lines)

                    # take the current set of text lines and add in the new one
                    new_lines = current_lines.values()
                    new_lines.append(text)


                    # uncomment below if you want to compare the new accuracy against the old
                    # if len(current_lines) > 1:
                    #     aligned_text = self.__get_aggregation_lines__(sorted_lines)
                    #     current_accuracy = self.__agreement__(aligned_text)
                    # else:
                    #     current_accuracy = -1

                    # what would the accuracy be if we added in this new user's line?
                    # new_lines = list(sorted_lines)
                    # assert isinstance(sorted_users,tuple)
                    # user_index = sorted_users.index(user)

                    # start by trying straight up replacing
                    # new_lines.append(text)
                    new_aligned = self.__line_alignment__(new_lines)
                    new_accuracy = self.__agreement__(new_aligned)
                    # todo - we can get two slightly different values for new_accuracy
                    # todo - because of slightly different approaches - is one better?
                    # todo - we might not need __agreement__, if not, we can remove it
                    # temp1,temp2,new_accuracy = self.__merge_aligned_text__(new_aligned)

                    # if the minimum accuracy resulted by adding in this line is still reasonably good
                    # add the line to the current cluster
                    if min(new_accuracy) >= 0.6:
                        current_pts[user] = (raw_pt,user)
                        current_lines[user] = text
                    else:
                        # otherwise, start a new cluster
                        clusters.append((current_lines.values(),current_pts.values()))
                        current_lines = {user:text}
                        current_pts = {user:(raw_pt,user)}

        # make sure to add the final cluster that we were working on at the end
        clusters.append((current_lines.values(),current_pts.values()))

        # remove any clusters which have only one user - treat those as noise
        for cluster_index in range(len(clusters)-1,-1,-1):
            # print len(clusters[cluster_index][0])
            if len(clusters[cluster_index][0]) <= 4: #2
                # assert len(clusters[cluster_index][1]) == 1
                clusters.pop(cluster_index)

        if len(clusters) == 0:
            return [],0

        # if we have more than one cluster - some of them might need to be merged
        # after removing "error" clusters
        # to do so - revert back to Hesse format
        # todo - maybe only run this if we have removed any error lines
        if len(clusters) > 1:

            hessen_lines = []

            for cluster_index in range(len(clusters)):
                lines_segments,users = zip(*clusters[cluster_index][1])
                x1_l, x2_l, y1_l, y2_l = zip(*lines_segments)
                x1,x2,y1,y2 = np.median(x1_l),np.median(x2_l),np.median(y1_l),np.median(y2_l)
                hessen_lines.append(hesse_line_reduction([[x1,x2,y1,y2],])[0])

            # print hessen_lines
            slope_l,angle_l = zip(*hessen_lines)
            max_s,min_s = max(slope_l),min(slope_l)
            max_a,min_a = max(angle_l),min(angle_l)

            # normalize values
            hessen_lines = [((max_s-s)/(max_s-min_s),(max_a-a)/(max_a-min_a)) for s,a in hessen_lines]

            tree = spatial.KDTree(hessen_lines)

            to_merge = []
            will_be_merged = set()

            for l_index in range(len(hessen_lines)-1,-1,-1):
                for l2_index in tree.query_ball_point(hessen_lines[l_index],0.15):
                    if l2_index > l_index:
                        t_lines = clusters[l_index][0][:]
                        t_lines.extend(clusters[l2_index][0])

                        aligned_text = self.__line_alignment__(t_lines)
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

            clusters = new_clusters

        # and now, finally, the actual text clustering
        cluster_centers = []
        cluster_pts = []
        cluster_users = []

        cluster_members = []


        agreement = []
        self.line_agreement.append([])

        for lines,pts_and_users in clusters:
            pts,users = zip(*pts_and_users)
            x1_values,x2_values,y1_values,y2_values = zip(*pts)

            # todo - handle when some of the coordinate values are not numbers -
            # todo - this corresponds to when there are multiple text segments from the same user
            # todo - this in turn corresponds to the case where we look for "broken" lines
            # todo - so definitely something down the line
            x1 = np.median(x1_values)
            x2 = np.median(x2_values)
            y1 = np.median(y1_values)
            y2 = np.median(y2_values)

            # align the text
            aligned_text = self.__line_alignment__(lines)

            # align the non-fasta version of the text lines
            nf_aligned_text = self.__add_alignment_spaces__(aligned_text,non_fasta_text,pts_and_users)

            # aggregate the lines - looking for character spots where there is mostly consensus
            aggregate_text,character_agreement,per_user_agreement = self.__merge_aligned_text__(nf_aligned_text)
            # print str(len(users)) + "\t" + aggregate_text

            aggregate_text = self.__reset_special_characters__(aggregate_text)
            # print "\t" + aggregate_text

            cluster_centers.append((x1,x2,y1,y2,aggregate_text))
            cluster_pts.append(zip(pts,lines))
            cluster_users.append(users)
            agreement.append(character_agreement)
            cluster_members.append(aligned_text)

            # use this if you want to keep track of stats
            # self.line_agreement[-1].append((character_agreement,len(users)))

        results = []
        for center,pts,users,lines,a in zip(cluster_centers,cluster_pts,cluster_users,cluster_members,agreement):
            results.append({"center":center,"cluster members":lines,"tools":[],"num users":len(users),"agreement":a})

        return results,0

    def __preliminarily__clustering__(self,markings,user_ids):
        """
        do an initial clustering based just on user ids and line coordinates (so ignore the text values)
        """
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

# def text_mapping(marking,image_dimensions):
#     # want to extract the params x1,x2,y1,y2 but
#     # ALSO make sure that x1 <= x2 and flip if necessary
#
#     x1 = marking["startPoint"]["x"]
#     x2 = marking["endPoint"]["x"]
#     y1 = marking["startPoint"]["y"]
#     y2 = marking["endPoint"]["y"]
#
#     try:
#         text = marking["text"]
#     except KeyError:
#         raise InvalidMarking(marking)
#
#     if x1 <= x2:
#         return x1,x2,y1,y2,text
#     else:
#         return x2,x1,y2,y1,text


def relevant_text_params(marking,image_dimensions):
    """
    extract the relevant params from the the transcription marking
    note that the text is the last item - which means we can treat the results
    pretty much like a line segment - which it mostly is
    :param marking:
    :param image_dimensions:
    :return:
    """
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
        AggregationAPI.__init__(self,"tate")#"tate",environment="staging")

        self.marking_params_per_shape = dict()

        self.marking_params_per_shape["text"] = relevant_text_params

        reduction_algs = {"text":text_line_reduction}
        self.__set_clustering_algs__({"text":TextCluster},reduction_algs)
        self.__set_classification_alg__(SubjectRetirement,{"host":self.host_api,"project_id":self.project_id,"token":self.token,"workflow_id":121})

        self.ignore_versions = True
        self.instructions[683] = {}

        self.only_retired_subjects = False
        self.only_recent_subjects = True

    def __get_workflow_details__(self,given_workflow_id=None):
        """
        override the basic aggregation_api call to get the details about workflows
        since tate does things differently
        :param given_workflow_id:
        :return:
        """
        workflows = dict()
        workflows[121] = self.__readin_tasks__(683)

        versions = dict()
        versions[121] = 17

        instructions = dict()
        updated_at_timestamps = {}

        return workflows,versions,instructions,updated_at_timestamps

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
        # pass
        # subject_id = int(project.__subject_ids_in_set__(905)[1])
        # subject_id = 603303
        # project.__migrate__()
        project.__aggregate__(subject_set=[662859])
        # project.__aggregate__(workflows=[121],subject_set=[subject_id])
        # print project.cluster_algs["text"].line_agreement
        # project.__migrate__()
        # project.__aggregate__(workflows=[121])
        # pickle.dump(project.cluster_algs["text"].line_agreement,open("/home/greg/agg.out","wb"))
        # ll = [l for (a,l) in project.cluster_algs["text"].line_agreement]
        # print max(ll),np.mean(ll),np.median(ll)
        # agreement_with_3 = [a for (a,l) in project.cluster_algs["text"].line_agreement if l >= 3]
        # print len(agreement_with_3)
        # print np.mean(agreement_with_3), np.median(agreement_with_3)
