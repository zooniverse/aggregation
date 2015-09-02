__author__ = 'ggdhines'
from simplified_transcription import SimplifiedTate
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from termcolor import colored

subject_id = 603303

with SimplifiedTate() as project:
    # for subject_id in project.__get_retired_subjects__(workflow_id):
    #     print subject_id
    #     for a in project.__cassandra_annotations__(workflow_id,[subject_id]):
    #         print a
    #         break

    project.classification_alg = None
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
    aggregated_text = project.__aggregate__(workflows=[workflow_id],subject_set=[subject_id],store_values=False)
    #
    lines = {}
    for id_,cluster in aggregated_text[subject_id]["T2"]["text clusters"].items():
        if id_ not in ["all_users","param"]:
            x1,x2,y1,y2,text = cluster["center"]
            lines[y1] = (cluster["cluster members"],text)
            plt.plot([x1,x2],[y1,y2],"o-",color="red")

    for y in sorted(lines.keys()):
        print y
        print lines[y][1]
        print "----===="
        for l in lines[y][0]:
            for ii,c in enumerate(l):
                if c != lines[y][1][ii]:
                    print "\b" + colored(c,"red"),
                else:
                    print "\b" + c,
            print
        print

    plt.show()