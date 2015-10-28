#!/usr/bin/env python
__author__ = 'ggdhines'
from transcription import Tate
import sys
from subprocess import call
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import re

latex_header = """
\documentclass[a4paper,10pt]{article}
%\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{color}
\usepackage[margin=0.25in]{geometry}
\usepackage{graphicx}
\\begin{document}


"""

folger_tags = {}

# folger_tags["<unclear>[.][*]</unclear>"] = "<unclear></unclear>"
folger_tags["<ex>"] = "<ex>"
folger_tags["</ex>"] = "</ex>"
folger_tags["<del>"] = "<del>"
folger_tags["</del>"] = "</del>"
folger_tags["<ins>"] = "<ins>"
folger_tags["</ins>"] = "</ins>"
folger_tags["<sup>"] = "<sup>"
folger_tags["</sup>"] = "</sup>"
folger_tags["<label>"] = "<label>"
folger_tags["</label>"] = "</label>"
folger_tags["<graphic>"] = "<graphic>"
folger_tags["</graphic>"] = "</graphic>"
folger_tags["<which></which>"] = "w<sw-ex>hi</sw-ex><sl>ch</sl>"
folger_tags["<with></with>"] = "w<sw-ex>i</sw-ex><sl>th</sl>"
folger_tags["<the></the>"] = "<brev-y>th</brev-y><sl>e</sl>"
folger_tags["<that></that>"] = "<brev-y>th</brev-y><sw-ex>a</sw-ex><sl>t</sl>"
folger_tags["<them></them>"] = "<brev-y>th</brev-y><sw-ex>e</sw-ex><sl>m</sl>"
folger_tags["<your></your>"] = "y<sw-ex>ou</sw-ex><sl>r</sl>"
folger_tags["<maiestie></maiestie>"] = "Ma<sw-ex>ies</sw-ex><sl>tie</sl>"
folger_tags["<worshipfull></worshipfull>"] = "Wor<sw-ex>shipfu</sw-ex><sl>ll</sl>"
folger_tags["<lady></lady>"] = "La<sw-ex>dy</sw-ex>"
folger_tags["<ladyship></ladyship>"] = "La<ex>dyshi</ex><sl>ll</sl>"
folger_tags["<lord></lord>"] = "L<sw-ex>ord</sw-ex>"
folger_tags["<lordship></lordship>"] = "L<ex>ordshi</ex>p"
folger_tags["<sir></sir>"] = "S<sw-ex>i</sw-ex><sl>r</sl>"
folger_tags["<our></our>"] = "o<sw-ex>u</sw-ex><sl>r</sl>"
folger_tags["<examinant></examinant>"] = "Exa<sw-ex>m</sw-ex>i<sw-ex>nan</sw-ex>te"
folger_tags["<item></item>"] = "It<sw-ex>e</sw-ex>m"
folger_tags["<letter></letter>"] = "l<sw-ex>ett</sw-ex>re"
folger_tags["<honorable></honorable>"] = "Ho<sw-ex>norable</sw-ex>"
folger_tags["<esquire></esquire>"] = "Esq<sw-ex>uire</sw-ex>"
folger_tags["<es></es>"] = "<brev-es>es</brev-es>"
folger_tags["<ment></ment>"] = "m<sw-ex>en</sw-ex><sl>t</sl>"
folger_tags["<paid></paid>"] = "p<sw-ex>ai</sw-ex><sl>d</sl>"
folger_tags["<Anno></Anno>"] = "A<sw-ex>nn</sw-ex><sl>o</sl>"

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
    for tag,new_tag in folger_tags.items():
        # text = re.sub(tag,new_tag,text)
        text = text.replace(tag,new_tag)

    if text.count("blue") > 5:
        print old_text
        # print intermediate_text
        print text
        assert False

    return text


project_id = 376
environment = "development"
with open("/tmp/transcription.tex","w") as f:
    f.write(latex_header)

    with Tate(project_id,environment) as project:
        for subject_id,aggregations in project.__yield_aggregations__(205):
            print subject_id

            metadata = project.__get_subject_metadata__(subject_id)["subjects"][0]["metadata"]
            print metadata
            print "file name" in metadata
            if "file name" in metadata:
                fname = metadata["file name"]
            else:
                fname = "subject id " + str(subject_id)

            lines = {}
            individual_lines = {}

            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            #
            image_fname = project.__image_setup__(subject_id)

            image_file = cbook.get_sample_data(image_fname)
            image = plt.imread(image_file)
            # fig, ax = plt.subplots()
            im = axes.imshow(image)
            #

            #


            empty = True

            num_users = []

            for key,line in aggregations["T2"]["text clusters"].items():
                if key in ["all_users","param"]:
                    continue

                num_users.append(line["num users"])

                x1,x2,y1,y2,text = line["center"]
                plt.plot([x1,x2],[y1,y2],"-",color="red",linewidth=0.5)
                lines[y1] = text

                pt_list,text_list = zip(*line["cluster members"])
                individual_lines[y1] = text_list

                empty = False




            line_items = lines.items()
            line_items.sort(key= lambda x:x[0])

            plt.axis('off')
            plt.savefig("/tmp/"+str(subject_id)+".pdf",bbox_inches='tight', pad_inches=0)
            plt.close()

            if not empty:

                f.write("\section{"+str(fname)+"}\n")
                f.write("\\begin{figure}[t]\centering \includegraphics[scale=1]{/tmp/"+str(subject_id)+".pdf} \end{figure}")

                for y,l in line_items:
                    cumulative_c = ""
                    for c in l:
                        if c == "&":
                            f.write(coloured_string(cumulative_c)+"\&")
                            cumulative_c = ""
                        elif ord(c) not in [24,27]:
                            # f.write(c)
                            cumulative_c += c
                        else:
                            f.write(coloured_string(cumulative_c)+"{\color{red}?}")
                            cumulative_c = ""

                    f.write(coloured_string(cumulative_c))
                    f.write("\\newline\n")

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
                        elif ord(c) not in [24,27]:
                            # f.write(c)
                            cumulative_c += c
                        else:
                            f.write(coloured_string(cumulative_c)+"{\color{red}?}")
                            cumulative_c = ""

                    f.write(coloured_string(cumulative_c))

                    f.write("\\newline\n--- \\newline\n")

                    for i_l in individual_lines[y]:
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
                            elif ord(c) not in [24,27]:
                                cumulative_c += c
                            else:
                                f.write(coloured_string(cumulative_c)+"{\color{red}-}")
                                cumulative_c = ""
                        f.write(coloured_string(cumulative_c))
                        f.write("}")
                        f.write("\\newline\n")

                    f.write("\\newline\n")
                f.write("\\newpage\n")


    f.write("\end{document}")

call(["pdflatex","-output-directory=/tmp","/tmp/transcription.tex"])
