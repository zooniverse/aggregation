__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from blob_clustering import BlobClustering
from agglomerative import Agglomerative
import json

with AggregationAPI(11,"development") as whales:
    whales.__setup__()
    # whales.__migrate__()

    rectangle_clustering = BlobClustering("rectangle",whales,{})
    point_clustering = Agglomerative("point",{})


    postgres_cursor = whales.postgres_session.cursor()
    select = "SELECT classification_subjects.subject_id from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84"
    postgres_cursor.execute(select)

    #

    for subject_id in postgres_cursor.fetchall():
        subject_id = subject_id[0]
        subject_id = 494953

        # T1 - rectangle outline
        # t2 - points
        _,markings,_,_ = whales.__sort_annotations__(84,[subject_id,])
        task_id = "T1"
        shape = "rectangle"


        select = "SELECT user_id,annotations from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84 and classification_subjects.subject_id = " + str(subject_id)
        postgres_cursor.execute(select)

        f_name = whales.__image_setup__(subject_id)
        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)
        # print json.dumps(markings[task_id]["rectangle"],indent=4, separators=(',', ': '))

        non_empty_markings = [(u,m,t) for u,m,t in markings[task_id][shape][subject_id] if m is not None]
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)
        if non_empty_markings != []:
            all_users,t1,t2 = zip(*markings[task_id][shape][subject_id])
            all_users = list(set(all_users))

            users,reduced_markings,tools = zip(*non_empty_markings)
            image_dimensions = {subject_id:image.shape}

            rect_markings = markings[task_id][shape][subject_id]
            cluster_results,time_to_cluster = rectangle_clustering.__cluster__(reduced_markings,users,tools,reduced_markings,image_dimensions[subject_id],subject_id)
            # x_l,y_l =  zip(*cluster_results[0]["center"][0])
            # print x_l
            # print y_l



            # rectangle_clustering.quad_root.__plot__(ax1)

            # print results
            # print cluster_results
            (x1,y1),(x2,y2) = cluster_results[0]["center"]
            plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1])

            # plt.show()

            for user_id,annotations in postgres_cursor.fetchall():
                print "==---"
                count = 0
                for whale in annotations[0]["value"]:
                    # if whale["details"] == [{u'value': None}]:
                    #     continue
                    count += 1
                    print user_id,whale

                    x1 = whale["x"]
                    y1 = whale["y"]
                    x2 = x1+whale["width"]
                    y2 = y1+whale["height"]

                    plt.plot(x1,y1,"o")
                    plt.plot(x1,y2,"o")
                    plt.plot(x2,y2,"o")
                    plt.plot(x2,y1,"o")

                # print count
            # print

        plt.ylim((image.shape[0],0))
        plt.xlim((0,image.shape[1]))
        plt.show()

        _, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)

        task_id = "T2"
        shape = "point"
        non_empty_markings = [(u,m,t) for u,m,t in markings[task_id][shape][subject_id] if m is not None]

        common_sets = []

        if non_empty_markings != []:
            all_users,t1,t2 = zip(*markings[task_id][shape][subject_id])
            all_users = list(set(all_users))

            users,reduced_markings,tools = zip(*non_empty_markings)
            image_dimensions = {subject_id:image.shape}

            pts_clusters = point_clustering.__cluster__(reduced_markings,users,tools,reduced_markings,image_dimensions[subject_id],subject_id)
            for p in pts_clusters[0]:
                x,y = p["center"]
            # for user_id,(x,y),_ in non_empty_markings:
                if p["num users"] > 3:
                    print sorted(p["users"])
                    common_sets.append(p["users"])
                    plt.plot(x,y,"o")
            plt.show()
        else:
            plt.close()

        for s1 in common_sets:
            for s2 in common_sets:
                print str(len([s for s in s1 if s in s2])) + " ",
            print

        _, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)

        task_id = "T3"
        shape = "point"
        non_empty_markings = [(u,m,t) for u,m,t in markings[task_id][shape][subject_id] if m is not None]
        for user_id,(x,y),_ in non_empty_markings:
            plt.plot(x,y,"o")
        plt.show()


        # break