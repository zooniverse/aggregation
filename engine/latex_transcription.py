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

            for key,line in aggregations["T2"]["text clusters"].items():
                if key in ["all_users","param"]:
                    continue

                x1,x2,y1,y2,text = line["center"]
                plt.plot([x1,x2],[y1,y2],"-",color="red",linewidth=0.5)
                lines[y1] = text

                empty = False

            line_items = lines.items()
            line_items.sort(key= lambda x:x[0])

            plt.axis('off')
            plt.savefig("/tmp/"+str(subject_id)+".pdf",bbox_inches='tight', pad_inches=0)
            plt.close()

            if not empty:
                print "***"
                f.write("\section{"+str(fname)+"}\n")
                f.write("\\begin{figure}[t]\centering \includegraphics[scale=0.95]{/tmp/"+str(subject_id)+".pdf} \end{figure}")


            for y,l in line_items:
                for c in l:
                    if c == "&":
                        f.write("\&")
                    elif ord(c) not in [24,27]:
                        f.write(c)
                    else:
                        f.write("{\color{red}?}")

                f.write("\\newline\n")

            if not empty:
                f.write("\\newpage\n")
    f.write("\end{document}")

call(["pdflatex","-output-directory=/tmp","/tmp/transcription.tex"])
