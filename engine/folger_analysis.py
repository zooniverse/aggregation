__author__ = 'ggdhines'
from transcription import Tate,TextCluster
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from termcolor import colored
import yaml
subject_id = 928459

with Tate(376,"development") as project:
    # for subject_id in project.__get_retired_subjects__(workflow_id):
    #     print subject_id
    #     for a in project.__cassandra_annotations__(workflow_id,[subject_id]):
    #         print a
    #         break

    api_details = yaml.load(open("/app/config/aggregation.yml","rb"))
    tag_file = api_details[376]["tags"]
    additional_clustering_args = {"tag_file":tag_file}
    clustering_alg = TextCluster("text",additional_clustering_args)

    project.classification_alg = None
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    workflow_id = 205
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
        aggregate_line =  lines[y][1]
        aggregate_string = []#aggregate_line.split("")

        reverse_map = {v: k for k, v in clustering_alg.tags.items()}
        reverse_map[27] = "_"

        for c in aggregate_line:
            if ord(c) in reverse_map:
                aggregate_string.append(reverse_map[ord(c)])
            else:
                aggregate_string.append(c)

        # # print aggregate_line
        # # print clustering_alg.__set_special_characters__(aggregate_line)
        #
        # for c in lines[y][1]:
        #     if ord(c) == 27:
        #         print "\b"+ colored("_","red"),
        #     else:
        #         print "\b"+c,
        # print

        individual_strings = []

        for l in zip(*lines[y][0])[1]:
            string = []
            for c in l:
                if ord(c) in reverse_map:
                    string.append(reverse_map[ord(c)])
                    print string
                else:
                    string.append(c)

            individual_strings.append(string)



        string_lengths = []
        for ii in range(len(aggregate_string)):
            m = max(len(aggregate_string[ii]),max([len(s[ii]) for s in individual_strings]))
            string_lengths.append(m)

        # print string_lengths

        # print out the aggregate
        for ii in range(len(aggregate_string)):
            c = aggregate_string[ii]
            if len(c) == 1 and ord(c) == 200:
                print "\b"+ colored("_","red"),
            else:
                print "\b"+c,

            for j in range(string_lengths[ii]-len(c)):
                print "\b ",
        print
        print "==="

        for s in individual_strings:
            for ii in range(len(s)):
                c = s[ii]
                if len(c) ==1 and ord(c) == 201:
                    c = chr(24)
                if c != aggregate_string[ii]:
                    print "\b"+ colored(c,"red"),
                else:
                    print "\b"+ c,

                for j in range(string_lengths[ii]-len(c)):
                    print "\b ",
            print
        print




    plt.show()