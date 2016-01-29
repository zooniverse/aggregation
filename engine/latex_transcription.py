#!/usr/bin/env python
__author__ = 'ggdhines'
from transcription import Tate
import sys
from subprocess import call
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import re
import yaml

latex_header = """
\documentclass[a4paper,10pt]{article}
%\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{color}
\usepackage[margin=0.25in]{geometry}
\usepackage{graphicx}
\\begin{document}


"""

def get_updated_tags(project_id):
    replacement_tags = {}
    param_file = open("/app/config/aggregation.yml","rb")
    param_details = yaml.load(param_file)

    if (project_id not in param_details) or ("tags" not in param_details[project_id]):
        # print "could not find any tag info in the yml file"
        assert False

    with open(param_details[project_id]["tags"],"rb") as f:
        for old_tag in f.readlines():
            old_tag = old_tag.strip()
            # print old_tag
            assert isinstance(old_tag,str)
            # old_tag = old_tag[:-1]
            if old_tag == "":
                break
            new_tag = old_tag.replace("sw-","")
            new_tag = new_tag.replace(".*","")

            # this is for annotate
            new_tag = new_tag.replace("\\","")
            # new_tag = new_tag.replace("\]","]")
            replacement_tags[old_tag] = new_tag
            # print "^^" +new_tag
            # print "==--"

    return replacement_tags

# project_id = 245

folger_tags = {}#get_updated_tags(376)


for old_tag,new_tag in folger_tags.items():
    folger_tags[old_tag] = "{\color{blue}" + new_tag + "}"


def coloured_string(text):
    old_text = text

    text = re.sub("<unclear>[.][*]</unclear>","{\color{blue}<unclear></unclear>}",text)

    # intermediate_dict = {}
    # second_intermediate_dict = {}
    # for ii,(tag,new_tag) in enumerate(folger_tags.items()):
    #     intermediate_dict[tag] = 200+ii
    #     second_intermediate_dict[200+ii] = new_tag

    # for tag,ii in intermediate_dict.items():
    #     text = re.sub(tag,chr(ii),text)
    #
    # intermediate_text = text
    #
    # for ii,new_tag in second_intermediate_dict.items():
    #     text = re.sub(chr(ii),new_tag,text)
    for tag,new_tag in sorted(folger_tags.items(), key = lambda x:len(x[0]),reverse=True):
        # text = re.sub(tag,new_tag,text)
        text = text.replace(tag,new_tag)

    if text.count("blue") > 5:
        # print old_text
        # print intermediate_text
        # print text
        assert False

    return text
import random

environment = "development"


def latex_output(project,workflow_id,subjects):
    folger_tags = get_updated_tags(245)

    # print folger_tags
    # assert False
    # assert False
    with open("/tmp/transcription.tex","w") as f:
        f.write(latex_header)

        # for count,(subject_id,aggregations) in enumerate(project.__yield_aggregations__(121,s)):
        for subject_id in subjects:#[671541,662859,649365,653381,672697]:
            print "** " + str(subject_id)
            aggregations = list(project.__yield_aggregations__(workflow_id,subject_id))
            if aggregations == []:
                continue
            aggregations = aggregations[0][1]
            # print subject_id

            metadata = project.__get_subject_metadata__(subject_id)["subjects"][0]["metadata"]
            if "file name" in metadata:
                fname = metadata["file name"]
            else:
                fname = "subject id " + str(subject_id)

            lines = {}
            individual_lines = {}

            empty = True

            num_users = []

            individual_pts = []

            # print aggregations

            for key,line in aggregations["T2"]["text clusters"].items():
                if key in ["all_users","param"]:
                    continue

                num_users.append(line["num users"])

                x1,x2,y1,y2,text = line["center"]
                # plt.plot([x1,x2],[y1,y2],"-",color="red",linewidth=0.5)
                lines[(x1,x2,y1,y2)] = text

                pt_list,text_list = zip(*line["cluster members"])
                individual_pts.extend(pt_list)
                individual_lines[(x1,x2,y1,y2)] = text_list

                empty = False

            line_items = lines.items()
            line_items.sort(key= lambda x:x[0][2])

            # if empty:
            #     plt.close()

            if not empty:
                fig = plt.figure()
                axes = fig.add_subplot(1, 1, 1)
                #
                image_fname = project.__image_setup__(subject_id)

                image_file = cbook.get_sample_data(image_fname)
                image = plt.imread(image_file)
                # fig, ax = plt.subplots()
                im = axes.imshow(image)
                plt.axis('off')


                f.write("\section{"+str(fname)+"}\n")
                f.write("\\begin{figure}[t]\centering \includegraphics[scale=1]{/tmp/"+str(subject_id)+".pdf} \end{figure}")

                for pts,l in line_items:
                    # print l
                    # print [ord(c) for c in l]
                    cumulative_c = ""

                    l = l.replace("$","\$")
                    l = l.replace("\\","\\textbackslash")
                    # l = l.replace("[","\[")
                    # l = l.replace("]","\]")
                    for c in l:
                        if c == "&":
                            f.write(coloured_string(cumulative_c)+"\&")
                            cumulative_c = ""
                        elif c == "%":
                            f.write(coloured_string(cumulative_c)+"\%")
                            cumulative_c = ""
                        elif ord(c) == 27:
                            if cumulative_c != "":
                                f.write(coloured_string(cumulative_c)+"{\color{red}?}")
                            else:
                                f.write("{\color{red}?}")
                            cumulative_c = ""
                        elif ord(c) != 24:
                            # f.write(c)
                            cumulative_c += c

                    plt.plot([pts[0],pts[1]],[pts[2],pts[3]],linewidth=0.5,color="red")

                    f.write(coloured_string(cumulative_c))
                    f.write("\\newline\n")

                t_ = project.__sort_annotations__(workflow_id,[subject_id])[1]
                # print individual_pts
                for ii,(user_id,transcription,tool) in enumerate(t_["T2"]["text"][subject_id]):
                    if transcription is None:
                        continue
                    coords = list(transcription[:-1])
                    # print coords
                    if not coords in individual_pts:
                        plt.plot([coords[0],coords[1]],[coords[2],coords[3]],linewidth=0.5,color="blue")
                # assert False

                plt.savefig("/tmp/"+str(subject_id)+".pdf",bbox_inches='tight', pad_inches=0,dpi = 500)
                plt.close()

                f.write("\\newline \\newline Number of transcriptions per line: " + str(num_users) + "\n")
                f.write("\\newpage\n")

                # now repeat for individual lines
                for y,l in line_items:
                    f.write("\\noindent ")
                    cumulative_c = ""
                    for c in l:
                        if c == "&":
                            f.write(coloured_string(cumulative_c)+"\&")
                            cumulative_c = ""
                        elif ord(c) == 27:
                            # we are not certain
                            f.write(coloured_string(cumulative_c)+"{\color{red}?}")
                            cumulative_c = ""
                        elif ord(c) != 24:
                            # f.write(c)
                            # 24 => gap, in which case we just skip
                            cumulative_c += c

                    f.write(coloured_string(cumulative_c))

                    f.write("\\newline\n--- \\newline\n")

                    for i_l in individual_lines[y]:
                        i_l = i_l.encode('ascii','ignore')
                        # i_l = i_l.decode('utf8')
                        # assert isinstance(i_l,unicode)
                        # i_l.replace('\u2014',"-")

                        # if "with" in i_l:
                        #     print i_l
                        #     assert False
                        f.write("\\textit{")
                        cumulative_c = ""
                        for c in i_l:

                            if c == "&":
                                f.write(coloured_string(cumulative_c)+"\&")
                                #f.write("\&")
                                cumulative_c = ""
                            elif c == "%":
                                f.write(coloured_string(cumulative_c)+"\%")
                                cumulative_c = ""
                            elif ord(c) not in [24,27]:
                                cumulative_c += c
                            else:
                                if cumulative_c != "":
                                    f.write(coloured_string(cumulative_c)+"{\color{red}-}")
                                else:
                                    f.write("{\color{red}-}")
                                cumulative_c = ""
                        try:
                            f.write(coloured_string(cumulative_c))
                        except UnicodeEncodeError:
                            print cumulative_c
                            print [(c,ord(c)) for c in cumulative_c]
                            raise

                        f.write("}")
                        f.write("\\newline\n")

                    f.write("\\newline\n")
                f.write("\\newpage\n")


        f.write("\end{document}")

    call(["pdflatex","-output-directory=/tmp","/tmp/transcription.tex"],stdout=open("/dev/null","wb"))
