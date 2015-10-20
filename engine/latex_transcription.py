__author__ = 'ggdhines'
from transcription import Tate
import sys
from subprocess import call
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

latex_header = """
\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{color}
\usepackage{graphicx}
\\begin{document}


"""
with open("/tmp/transcription.tex","w") as f:
    f.write(latex_header)

    with Tate(sys.argv[1],sys.argv[2]) as project:
        for subject_id,aggregations in project.__yield_aggregations__(121):
            f.write("\section{"+str(subject_id)+"}\n")
            lines = {}

            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            workflow_id = 121
            #
            image_fname = project.__image_setup__(subject_id)

            image_file = cbook.get_sample_data(image_fname)
            image = plt.imread(image_file)
            # fig, ax = plt.subplots()
            im = axes.imshow(image)
            #

            #


            for key,line in aggregations["T2"]["text clusters"].items():
                if key in ["all_users","param"]:
                    continue

                x1,x2,y1,y2,text = line["center"]
                plt.plot([x1,x2],[y1,y2],"o-",color="red")
                lines[y1] = text

            line_items = lines.items()
            line_items.sort(key= lambda x:x[0])

            plt.savefig("/tmp/"+str(subject_id)+".pdf")

            f.write("\\begin{figure}[t]\centering \includegraphics[scale=0.75]{/tmp/"+str(subject_id)+".pdf} \end{figure}")


            for y,l in line_items:
                for c in l:
                    if ord(c) not in [24,27]:
                        f.write(c)

                f.write("\\newline\n")
    f.write("\end{document}")

call(["pdflatex","-output-directory=/tmp","/tmp/transcription.tex"])
