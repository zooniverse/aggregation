from __future__ import print_function
import re
import clustering
import unicodedata
import random
import os
import tempfile
from helper_functions import warning

__author__ = 'ggdhines'

def levenshtein(a,b):
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


class TextClustering(clustering.Cluster):
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

        aligned_text = []

        if len(lines) == 1:
            return lines


        with tempfile.NamedTemporaryFile(suffix=".fasta") as in_file, tempfile.NamedTemporaryFile("r") as out_file:
            for line in lines:
                if isinstance(line,tuple):
                    # we have a list of text segments which we should join together
                    line = "".join(line)

                # line = unicodedata.normalize('NFKD', line).encode('ascii','ignore')
                # assert isinstance(line,str)

                # for i in range(max_length-len(line)):
                #     fasta_line += "-"

                try:
                    in_file.write(">\n"+line+"\n")
                except UnicodeEncodeError:
                    warning(line)
                    warning(unicodedata.normalize('NFKD', line).encode('ascii','ignore'))
                    raise
            in_file.flush()

            # todo - play around with gap penalty --op 0.5
            t = "mafft --op 0.85 --text " + in_file.name + " > " + out_file.name +" 2> /dev/null"
            # t = "mafft --text " + in_file.name + " > " + out_file.name +" 2> /dev/null"

            os.system(t)

            cumulative_line = ""
            for line in out_file.readlines():
                if (line == ">\n"):
                    if (cumulative_line != ""):
                        aligned_text.append(cumulative_line)
                        cumulative_line = ""
                else:
                    cumulative_line += line[:-1]

            if cumulative_line == "":
                warning(lines)
                assert False
            aligned_text.append(cumulative_line)

        # no idea why mafft seems to have just including this line in the output
        # also might just be affecting Greg's computer
        if aligned_text[0] == '/usr/lib/mafft/lib/mafft':
            return aligned_text[1:]
        else:
            return aligned_text

    def __accuracy__(self,s):
        assert isinstance(s,str)
        assert len(s) > 0
        return sum([1 for c in s if c != "-"])/float(len(s))

    def __set_tags__(self,text):
        # convert to ascii
        try:
            text = text.encode('ascii','ignore')
        except AttributeError:
            warning(text)
            raise

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