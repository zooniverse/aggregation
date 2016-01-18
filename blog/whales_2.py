__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from blob_clustering import BlobClustering

with AggregationAPI(11,"development") as whales:
    whales.__setup__()
    # whales.__migrate__()

    rectangle_clustering = BlobClustering("rectangle",whales,{})


    postgres_cursor = whales.postgres_session.cursor()
    select = "SELECT classification_subjects.subject_id from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84"
    postgres_cursor.execute(select)

    #

    for subject_id in postgres_cursor.fetchall():
        subject_id = (494953,)
        annotations = whales.__sort_annotations__(84,list(subject_id))






        select = "SELECT annotations from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84 and classification_subjects.subject_id = " + str(subject_id[0])
        postgres_cursor.execute(select)

        f_name = whales.__image_setup__(subject_id[0])
        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)

        results = rectangle_clustering.__aggregate__(annotations[1],{subject_id[0]:image.shape})

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)

        p1,p2 = results[subject_id[0]]["T1"]["rectangle clusters"][0]["center"]

        plt.plot(p1[0],p1[1],"o")
        plt.plot(p2[0],p2[1],"o")
        plt.show()
        break

        for annotations in postgres_cursor.fetchall():

            count = 0
            for whale in annotations[0][0]["value"]:
                # if whale["details"] == [{u'value': None}]:
                #     continue
                count += 1

                x1 = whale["x"]
                y1 = whale["y"]
                x2 = x1+whale["width"]
                y2 = y1+whale["height"]

                plt.plot(x1,y1,"o")
                plt.plot(x1,y2,"o")
                plt.plot(x2,y2,"o")
                plt.plot(x2,y1,"o")

            print count
        print

        plt.ylim((image.shape[0],0))
        plt.xlim((0,image.shape[1]))
        plt.show()
        # break