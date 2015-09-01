__author__ = 'ggdhines'
from simplified_transcription import SimplifiedTate
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

subject_id = 603260

with SimplifiedTate() as project:
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    workflow_id = 121

    image_fname = project.__image_setup__(603260)

    image_file = cbook.get_sample_data(image_fname)
    image = plt.imread(image_file)
    # fig, ax = plt.subplots()
    im = axes.imshow(image)

    aggregated_text = project.__aggregate__(workflows=[workflow_id],subject_set=[603260],store_values=False)

    for id_,cluster in aggregated_text[subject_id]["T2"]["text clusters"].items():
        if id_ not in ["all_users","param"]:
            x1,x2,y1,y2,text = cluster["center"]
            print cluster
            plt.plot([x1,x2],[y1,y2],"o-",color="red")

    plt.show()

    # for subject_id in project.__get_retired_subjects__(workflow_id):
    # for subject_id in [603260]:
    #     # print subject_id
    #     # for a in project.__cassandra_annotations__(workflow_id,[subject_id]):
    #     #     print a
    #     # print
    #
    #
    #     # break