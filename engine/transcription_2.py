#!/usr/bin/env python
# import matplotlib
# matplotlib.use('WXAgg')
from aggregation_api import AggregationAPI
import helper_functions
from classification import Classification
import clustering
import numpy as np
import re
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
import botocore
import matplotlib.pyplot as plt
import math
import tempfile
import networkx
import botocore.session
import tarfile

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

__author__ = 'greg'

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"


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
        self.folger_safe_tags = dict()
        tag_counter = 149

        if "tags" in param_dict:
            with open(param_dict["tags"],"rb") as f:
                for l in f.readlines():
                    tag = l[:-1]
                    assert isinstance(tag,str)
                    self.tags[tag_counter] = tag
                    self.folger_safe_tags[tag_counter] = tag.replace("sw-","")
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
        assert len(lines) > 1

        aligned_text = []

        # with open(base_directory+"/Databases/transcribe"+id_+".fasta","wb") as f:
        with tempfile.NamedTemporaryFile(suffix=".fasta") as f_fasta, tempfile.NamedTemporaryFile() as f_out:
            for line in lines:
                try:
                    f_fasta.write(">\n"+line+"\n")
                except UnicodeEncodeError:
                    print line
                    print unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                    raise
            f_fasta.flush()
            # todo - play around with gap penalty --op 0.5
            t = "mafft  --text " + f_fasta.name + " > " + f_out.name + " 2> /dev/null"
            os.system(t)

            cumulative_line = ""
            for line in f_out.readlines():
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

        # the order of the keys matters - we need them to constant across all uses cases
        # we could sort .items() but that would be a rather large statement
        # replace each tag with a single non-standard ascii character (given by chr(num) for some number)
        text = text.strip()

        for chr_representation in sorted(self.tags.keys()):
            tag = self.tags[chr_representation]

            text = re.sub(tag,chr(chr_representation),text)

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
        "A" is used to represent all tags (all for an abitrary number of tags)
        also return a string which capitalization kept in tack and only the special tags removed
        so going from lower_text to text, allows us to recover what the captialization should have been
        """

        # lower text is what we will give to MAFFT - it can contain upper case letters but those will
        # all represent something special, e.g. a tag
        lower_text = text.lower()

        # for lower_text, every tag will be represented by "A" - MAFFT cannot handle characters with
        # a value of greater than 127. To actually determine which tag we are talking about
        # will have to refer to text
        # strings are immutable in Python so we have to rebuild from scratch
        new_lower_text = ""
        for i,c in enumerate(lower_text):
            if ord(c) > 127:
                new_lower_text += "A"

            else:
                new_lower_text += c
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

        return text,lower_text

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
                ret_text += self.folger_safe_tags[ord(c)]
            else:
                ret_text += c

        assert isinstance(text,str)
        return ret_text

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

        # self.stats["line_length"].append(len(aligned_text[0]))

        vote_history = []

        for char_index in range(len(aligned_text[0])):
            # get all the possible characters
            # todo - we can reduce this down to having to loop over each character once
            # todo - handle case (lower case vs. upper case) better
            char_set = set(text[char_index] for text in aligned_text)
            # get the percentage of votes for each character at this position
            char_vote = {c:sum([1 for text in aligned_text if text[char_index] == c]) for c in char_set if ord(c) != 25}
            vote_history.append(char_vote)

            # get the most common character (also the most likely to be the correct one) and the percentage of users
            # who "voted" for it

            # have at least 3 people transcribed this character?
            if sum(char_vote.values()) >= 3:
                most_likely_char,max_votes = max(char_vote.items(),key=lambda x:x[1])
                vote_percentage = max_votes/float(sum(char_vote.values()))

                # is there general agreement about what this character is?
                if vote_percentage > 0.5:
                    num_agreed += 1
                    aggregate_text += most_likely_char

                # check for special cases with double spaces or only differences about capitalization
                elif len(char_vote) == 2:
                    sorted_keys = [c for c in sorted(char_vote.keys())]
                    # this case => at least one person transcribed " " and at least one other
                    # person transcribed 24 (i.e. nothing) - so it might be that the first person accidentally
                    # gave " " which we'll assume means a double space so skip
                    if (ord(sorted_keys[0]) == 24) and (sorted_keys[1] == " "):
                        # but only skip it if at least two person gave 24
                        raw_counts = {c:sum([1 for text in aligned_text if text[char_index] == c]) for c in char_set}
                        if raw_counts[chr(24)] >= 2:
                            aggregate_text += chr(24)
                        else:
                            # 27 => disagreement
                            aggregate_text += chr(27)

                    # capitalization issues? only two different transcriptions given
                    # one the lower case version of the other
                    elif sorted_keys[0].lower() == sorted_keys[1].lower():
                        aggregate_text += sorted_keys[0].upper()
                    # otherwise two different transcriptions but doesn't meet either of the special cases
                    else:
                        aggregate_text += chr(27)
                else:
                    # chr(27) => disagreement
                    aggregate_text += chr(27)
            else:
                # not enough people have transcribed this character
                aggregate_text += chr(26)

        assert len(aggregate_text) > 0

        try:
            percent_consensus = num_agreed/float(len([a for a in aggregate_text if ord(a) != 26]))
            percent_complete = len([a for a in aggregate_text if ord(a) != 26])/float(len(aggregate_text))
        except ZeroDivisionError:
            percent_complete = 0
            percent_consensus = -1

        return aggregate_text,percent_consensus,percent_complete

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

            # added spaces before or after all of the text need to be treated differently
            non_space_characters = [i for (i,c) in enumerate(text) if c != "-"]
            try:
                first_char = min(non_space_characters)
            except ValueError:
                print text
                print nf_text
                print aligned_text_list
                raise
            last_char = max(non_space_characters)

            i = 0
            for j,c in enumerate(text):
                if c == "-":
                    if first_char <= j <= last_char:
                        aligned_nf_text += chr(24)
                    else:
                        aligned_nf_text += chr(25)
                else:
                    aligned_nf_text += nf_text[i]
                    i += 1
            aligned_nf_text_list.append(aligned_nf_text)

        return aligned_nf_text_list

    def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions,subject_id,recursive=False):
        """
        cluster the line segments transcriptions together - look for overlaping parts
        note that overlaping is not transitive - if A overlaps B and B overlap C, it does not follow
        that A overlaps C. So we'll use some graph theory instead to search for
        """

        G = networkx.Graph()
        G.add_nodes_from(range(len(markings)))

        # image is kept mainly just for debugging
        image = None

        # print len(markings)

        pruned_markings = []

        removed_count = 0

        # throw out any non-horizontal lines or those with non-ascii transcriptions
        for m_i,(x1,x2,y1,y2,t) in enumerate(markings):
            # skip empty strings - but make sure when checking to first remove tags that shouldn't
            # be there in the first place
            # set_tags removes some tags (such as <br>) which we don't want at all
            # so if a transcription is just "<br>" we should skip it
            processed_text = self.__set_tags__(t.encode('ascii','ignore'))
            if processed_text == "":
                continue

            # check the angle
            try:
                tan_theta = math.fabs(y1-y2)/math.fabs(x1-x2)
                theta = math.atan(tan_theta)
            except ZeroDivisionError:
                theta = math.pi/2.

            if math.fabs(theta) > 0.1:
                removed_count += 1
                continue

            pruned_markings.append((x1,x2,y1,y2,processed_text))

        # now look for the overlapping parts
        # examine every pair - note that distance from A to B does not necessarily equal
        # the distance from B to A - so order matters
        for m_i,(x1,x2,y1,y2,t) in enumerate(pruned_markings):
            for m_i2,(x1_,x2_,y1_,y2_,_) in enumerate(pruned_markings):
                # assuming two purely horizontal lines - consider the following example
                # x1 ----------- x2
                #     x1_----x2_
                # here the distance from x1 to the second line is the distance from x1 to x1_
                # but the distance from x1_ to the first line, is purely the vertical distance
                # since ignoring vertical distance, the second line is a subset of the first
                # so we can be pretty sure that the second line transcribes a subset of the first
                if m_i == m_i2:
                    continue

                # since we threw out all non-horizontal lines (within a certain degree of error)
                # the slope doesn't really matter - but doesn't hurt and should help when we generalize to
                # non-horizontal lines
                slope = (y2_-y1_)/float(x2_-x1_)
                inter = y2_ - slope*x2_

                # see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                # for explanation of the below code
                a = -slope
                b = 1
                c = -inter
                dist_1 = math.fabs(a*x1+b*y1+c)/math.sqrt(a**2+b**2)
                x = (b*(b*x1-a*y1)-a*c)/float(a**2+b**2)

                # if x is outside of the line segment in either direction - recalculate the distance as explained
                # above
                # todo - we could probably merge this if statement with the next one
                if x < x1_:
                    x = x1_
                    y = y1_
                    dist_1 = math.sqrt((x-x1)**2+(y-y1)**2)
                elif x > x2_:
                    x = x2_
                    y = y2_
                    dist_1 = math.sqrt((x-x1)**2+(y-y1)**2)

                # repeat for x2,y2
                dist_2 = math.fabs(a*x2+b*y2+c)/math.sqrt(a**2+b**2)
                x = (b*(b*x2-a*y2)-a*c)/float(a**2+b**2)

                if x < x1_:
                    x = x1_
                    y = y1_
                    dist_2 = math.sqrt((x-x2)**2+(y-y2)**2)
                elif x > x2_:
                    x = x2_
                    y = y2_
                    dist_2 = math.sqrt((x-x2)**2+(y-y2)**2)

                # if the average distance is less than 10 (an arbitrary threshold) then assume there is an overlap
                # that is - these two transcriptions are transcribing at least some of the same text
                if (dist_1+dist_2)/2. < 10:
                    G.add_path([m_i,m_i2])

        # look for connect components - i.e. sets of overlapping transcriptions
        clusters = [c for c in list(networkx.connected_components(G)) if len(c) > 1]

        # colours are just for debugging stuff
        colours = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))

        return_value_clusters = []

        for c,col in zip(clusters,colours):
            # deal with tags (make them all the same length) and any characters that MAFFT can't handle
            text_items = [self.__set_special_characters__(pruned_markings[i][-1]) for i in c]
            # original_items are the text with each tag represented by a single number taking up just one
            # character
            original_items,lowercase_items = zip(*text_items)

            # align based on the lower case items (such strings can contain upper case letters but they represent
            # something special)
            aligned_text = self.__line_alignment__(lowercase_items)
            # use the alignment spaces to align the original text
            aligned_uppercase_text = self.__add_alignment_spaces__(aligned_text,original_items)

            # finally aggregate the individual pieces of text together
            aggregate_text,percent_consensus,percent_complete = self.__merge_aligned_text__(aligned_uppercase_text)

            new_cluster = {}
            # get the average starting and ending points
            # todo - multiple small transcriptions are going to cause problems
            x1_list,x2_list,y1_list,y2_list,_ = zip(*[pruned_markings[i] for i in c])
            x1 = np.median(x1_list)
            x2 = np.median(x2_list)
            y1 = np.median(y1_list)
            y2 = np.median(y2_list)

            # aggregate_text still has whole tag "sets" represented by one individual character with ord() > 128
            # which postgres can't handle in text. So we need to replace each of those characters with its original
            # tag "set" ("set" because for folger, we can have a whole bunch of tags)
            # note for Folger - this IS where we convert to the tag format that Folger wants
            # also things like using "I" to represent " " should have been taken care of when we started using
            # aligned_uppercase_text
            aggregate_text = self.__reset_tags__(aggregate_text)

            new_cluster["center"] = (x1,x2,y1,y2,aggregate_text)

            new_cluster["cluster members"] = []
            for i in c:
                text = pruned_markings[i][-1]
                original_text = self.__reset_tags__(text)
                new_cluster["cluster members"].append((pruned_markings[i][:-1],original_text))

            new_cluster["num users"] = len(new_cluster["cluster members"])

            return_value_clusters.append(new_cluster)

        return return_value_clusters,0




    # def __create_cluster__(self,markings,index_filter):
    #     markings_in_cluster = [markings[i] for i in index_filter]
    #
    #     x1_values,x2_values,y1_values,y2_values,transcriptions = zip(*markings_in_cluster)
    #
    #     transformed_transcriptions = [self.__set_special_characters__(t) for t in transcriptions]
    #     # in lower case transformed, all of the original characters have been put into lower case
    #     # and upper case are used for special characters
    #     # we'll use lower_case_transformed to align the strings and by mapping back to capitalized_transformed
    #     # we can recover the correct capitalization for each string
    #     lower_case_transformed,capitalized_transformed = zip(*transformed_transcriptions)
    #
    #     aligned_transcriptions = self.__line_alignment__(lower_case_transformed)
    #     # recover the correct capitalizations
    #     aligned_transcriptions = self.__add_alignment_spaces__(aligned_transcriptions,capitalized_transformed)
    #
    #     # in cases where there is disagreement, use voting to determine the most likely character
    #     # if there is strong disagreement, we'll mark those spots as unknown
    #     aggregate_text,character_agreement,per_user_agreement = self.__merge_aligned_text__(aligned_transcriptions)
    #
    #     aggregate_text = self.__reset_special_characters__(aggregate_text)
    #
    #     x1 = np.median(x1_values)
    #     x2 = np.median(x2_values)
    #     y1 = np.median(y1_values)
    #     y2 = np.median(y2_values)
    #
    #     cluster = {}
    #     cluster["center"] = (x1,x2,y1,y2,aggregate_text)
    #
    #     cluster["tools"] = []
    #
    #     cluster["cluster members"] = []
    #     for ii,m in enumerate(markings_in_cluster):
    #         coords = m[:-1]
    #         text = self.__reset_special_characters__(aligned_transcriptions[ii])
    #         cluster["cluster members"].append((coords,text))
    #
    #     cluster["num users"] = len(cluster["cluster members"])
    #
    #     if cluster["num users"] >= 3:
    #         self.stats["retired lines"] += 1
    #
    #         aggregate_text = cluster["center"][-1]
    #
    #         errors = sum([1 for c in aggregate_text if ord(c) == 27])
    #         self.stats["errors"] += errors
    #         self.stats["characters"] += len(aggregate_text)
    #
    #
    #     return cluster

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
    #     of the transcriptions in that cluster - so no text aggregation/alignment is actually happening here
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
    #
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
    #         for ii,d in distances[1:20]:
    #             if ii not in assigned_to_cluster:
    #                 if text[ii] == "":
    #                     continue
    #
    #                 # create a new temp. set of transcriptions by adding in this next transcription
    #                 temp_transcriptions = transcriptions[:]
    #                 temp_transcriptions.append(text[ii])
    #
    #                 # check to see what the minimum accuracy is
    #                 # if high, go with this new set of transcriptions
    #                 # otherwise - allow one skip
    #                 aligned_transcriptions = self.__line_alignment__(temp_transcriptions)
    #                 accuracy = self.__agreement__(aligned_transcriptions)
    #
    #                 if min(accuracy) >= 0.6:
    #                     transcriptions = temp_transcriptions
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

        self.overall_aggregation = None
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
        cluster_aggregation = self.text_algorithm.__aggregate__(raw_markings,image_dimensions)
        image_aggregation = self.image_algorithm.__aggregate__(raw_markings,image_dimensions)

        self.overall_aggregation = self.__merge_aggregations__(cluster_aggregation,image_aggregation)

        return self.overall_aggregation

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

        self.text_algorithm = TextCluster("text",self,additional_text_args)
        self.image_algorithm = BlobClustering("image",self,{})

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

    # def __add_meta_data__(self):

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
        new_time_string = self.end_date.strftime("%B %d %Y")

        if float(stats["characters"]) == 0:
            accuracy = -1
        else:
            accuracy =  1. - stats["errors"]/float(stats["characters"])

        print stats

        subject = "Aggregation summary for " + str(old_time_string) + " to " + str(new_time_string)

        body = "This week we have retired " + str(num_retired) + " subjects, of which " + str(non_blanks_retired) + " where not blank."
        body += " A total of " + str(stats["retired lines"]) + " lines were retired. "
        body += " The accuracy of these lines was " + "{:2.1f}".format(accuracy*100) + "% - defined as the percentage of characters where at least 3/4's of the users were in agreement. A negative value indicates nothing was retired. \n\n"

        # if a path has been provided to the tar results, upload them to s3 and create a signed link to them
        if tar_path is not None:
            # just to maintain some order - store the results in the already existing tar file created for this
            # project - can cause trouble if we have never asked for the aggregation results via the PFE before
            aggregation_export_summary = self.__panoptes_call__("projects/"+str(self.project_id)+"/aggregations_export?admin=true")
            aws_bucket = aggregation_export_summary["media"][0]["src"]

            aws_fname,_ = aws_bucket.split("?")
            fname = aws_bucket.split("/")[-1]

            # create the actual connection to s3 and upload the file
            s3 = boto3.resource('s3')
            key_base = "panoptes-uploads.zooniverse.org/production/project_aggregations_export/"
            data = open(tar_path,"rb")
            bucket_name = "zooniverse-static"
            bucket = s3.Bucket(bucket_name)


            bucket.put_object(Key=key_base+fname, Body=data)

            # now create the signed link
            # results_key = bucket.get_key(fname)
            # results_url = results_key.generate_url(3600, query_auth=False, force_http=True)

            session = botocore.session.get_session()
            client = session.create_client('s3')
            presigned_url = client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': key_base+fname},ExpiresIn = 3600)

            body += "The aggregation results can found at " + presigned_url

        body += "\n Greg Hines \n Zooniverse \n \n PS The above link may contain a zip file within a zip file - I'm working on that."

        client = boto3.client('ses')
        response = client.send_email(
            Source='greg@zooniverse.org',
            Destination={
                'ToAddresses': [
                    'greg@zooniverse.org'#,'victoria@zooniverse.org','matt@zooniverse.org'
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

        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200



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

    if summary:
        assert end_date is not None

    with TranscriptionAPI(project_id,environment,end_date) as project:
        project.__setup__()
        #project.__migrate__()
        print "done migrating"

        project.__aggregate__()

        if summary:

            tar_path = "/tmp/"+str(project_id)+".tar.gz"
            t = tarfile.TarFile(tar_path,"w")
            json.dump(project.overall_aggregation,open("/tmp/"+str(project_id)+".json","wb"))
            t.add("/tmp/"+str(project_id)+".json")

            project.__summarize__(tar_path)

        # print aggregated_text