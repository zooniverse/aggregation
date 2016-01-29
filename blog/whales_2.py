#!/usr/bin/env python
import matplotlib
matplotlib.use('WXAgg')
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from blob_clustering import BlobClustering
from agglomerative import Agglomerative
import numpy as np
import math

__author__ = 'ggdhines'

def count_white(image,threshold):
    inds_0 = image[:,:,0] > threshold
    inds_1 = image[:,:,1] > threshold
    inds_2 = image[:,:,2] > threshold
    inds = inds_0 & inds_1 & inds_2
    # image[inds_dark] = [0,0,0]

    # inds = image[:,:,2] >= 50
    # image[inds] = [0,0,0]
    y,x = np.where(inds)

    return len(zip(x,y))/float(image.shape[0]*image.shape[1])

    # fig, ax1 = plt.subplots(1, 1)
    # ax1.imshow(image)
    # plt.show()

user_buckets = {5:0.05,4:0.25,3:0.5,2:0.75,1:0.95}
alg_threshold = range(100,250,10)

results = {t:[] for t in alg_threshold}
user_results = []

with AggregationAPI(11,"development") as whales:
    whales.__setup__()
    # whales.__migrate__()

    rectangle_clustering = BlobClustering("rectangle",whales,{})
    point_clustering = Agglomerative("point",{})


    postgres_cursor = whales.postgres_session.cursor()
    select = "SELECT classification_subjects.subject_id from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84"
    postgres_cursor.execute(select)

    #

    for subject_id in postgres_cursor.fetchall()[:50]:
        subject_id = subject_id[0]
        print subject_id
        # subject_id = 494953

        # T1 - rectangle outline
        # t2 - points
        classifications,markings,_,_ = whales.__sort_annotations__(84,[subject_id,])
        task_id = "T1"
        shape = "rectangle"

        f_name = whales.__image_setup__(subject_id)
        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)
        # print json.dumps(markings[task_id]["rectangle"],indent=4, separators=(',', ': '))

        non_empty_markings = [(u,m,t) for u,m,t in markings[task_id][shape][subject_id] if m is not None]
        # fig, ax1 = plt.subplots(1, 1)
        # ax1.imshow(image)

        if non_empty_markings != []:
            if "T1_0_0" not in classifications:
                continue

            # print classifications["T1_0_0"][subject_id].values()

            pruned = [c for c in classifications["T1_0_0"][subject_id].values() if c is not None ]
            # pruned = [v for v in pruned_c if v is not None]
            if pruned == []:
                continue
            # print "user classification is " + str(int(round(np.median(pruned)))+1)
            user_percent = user_buckets[int(round(np.median(pruned)))+1]
            user_results.append(user_percent)

            all_users,t1,t2 = zip(*markings[task_id][shape][subject_id])
            all_users = list(set(all_users))

            users,reduced_markings,tools = zip(*non_empty_markings)
            image_dimensions = {subject_id:image.shape}

            rect_markings = markings[task_id][shape][subject_id]
            cluster_results,time_to_cluster = rectangle_clustering.__cluster__(reduced_markings,users,tools,reduced_markings,image_dimensions[subject_id],subject_id)

            # print results
            # print cluster_results
            try:
                (x1,y1),(x2,y2) = cluster_results[0]["center"]


            except:
                continue

            # plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1])

            # plt.show()
            # plt.close()
            x_t = max(x1,x2)+1
            x_b = min(x1,x2)
            y_b = min(y1,y2)
            y_t = max(y1,y2)+1

            for t in alg_threshold:
                alg_percent = count_white(image[y_b:y_t,x_b:x_t,:],t)
                results[t].append(alg_percent)

    error = []
    for t in alg_threshold:
        diff = sum([(u-r)**2 for (u,r) in zip(user_results,results[t])])
        error.append(math.sqrt(diff/float(len(user_results))))

    plt.plot(alg_threshold,error)
    plt.show()

        # plt.plot(user_results,results[t],".")
    # plt.show()

        #     for user_id,annotations in postgres_cursor.fetchall():
        #         # print "==---"
        #         count = 0
        #         for whale in annotations[0]["value"]:
        #             # if whale["details"] == [{u'value': None}]:
        #             #     continue
        #             count += 1
        #             # print user_id,whale
        #
        #             x1 = whale["x"]
        #             y1 = whale["y"]
        #             x2 = x1+whale["width"]
        #             y2 = y1+whale["height"]
        #
        #             plt.plot(x1,y1,"o")
        #             plt.plot(x1,y2,"o")
        #             plt.plot(x2,y2,"o")
        #             plt.plot(x2,y1,"o")
        #
        #
        #
        # plt.ylim((image.shape[0],0))
        # plt.xlim((0,image.shape[1]))
        # plt.show()
        #
        # _, ax1 = plt.subplots(1, 1)
        # ax1.imshow(image)
        #
        # task_id = "T2"
        # shape = "point"
        # non_empty_markings = [(u,m,t) for u,m,t in markings[task_id][shape][subject_id] if m is not None]
        #
        # common_sets = []
        #
        # if non_empty_markings != []:
        #     all_users,t1,t2 = zip(*markings[task_id][shape][subject_id])
        #     all_users = list(set(all_users))
        #
        #     users,reduced_markings,tools = zip(*non_empty_markings)
        #     image_dimensions = {subject_id:image.shape}
        #
        #     pts_clusters = point_clustering.__cluster__(reduced_markings,users,tools,reduced_markings,image_dimensions[subject_id],subject_id)
        #     for p in pts_clusters[0]:
        #         x,y = p["center"]
        #
        #         if p["num users"] > 3:
        #             print sorted(p["users"])
        #             common_sets.append(p["users"])
        #             plt.plot(x,y,"o")
        #     plt.show()
        # else:
        #     plt.close()
        #
        # for s1 in common_sets:
        #     for s2 in common_sets:
        #         print str(len([s for s in s1 if s in s2])) + " ",
        #     print
        #
        # _, ax1 = plt.subplots(1, 1)
        # ax1.imshow(image)
        #
        # task_id = "T3"
        # shape = "point"
        # non_empty_markings = [(u,m,t) for u,m,t in markings[task_id][shape][subject_id] if m is not None]
        # for user_id,(x,y),_ in non_empty_markings:
        #     plt.plot(x,y,"o")
        # plt.show()
        #
        #
        # # break