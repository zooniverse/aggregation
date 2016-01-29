__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
# from sklearn.cluster import KMeans
# import matplotlib.cbook as cbook
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
import matplotlib.cbook as cbook
import math
import random
import itertools

def get_user_pts(markings):
    user_pts = []
    X = np.asarray(markings)
    db = DBSCAN(eps=10, min_samples=3).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print n_clusters_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        user_pts.append(xy)
        x,y = zip(*xy)
        # plt.plot(x,y,"o")


    return user_pts

def correction(f_name,pts):
    m = []
    plt.close()

    image_file = cbook.get_sample_data(f_name)
    image = plt.imread(image_file)

    good_parallelograms = []

    ds = []

    for s in itertools.combinations(range(len(pts)), 4):
        subset = [pts[i] for i in s]
        # print s

        centers = []
        for p in subset:
            x,y = zip(*p)
            centers.append([np.median(x),np.median(y)])

        # try to reconstruct a rectangle given this start point
        A = 0
        B_dist = float("inf")
        D_dist = float("inf")
        D = None
        B = None
        # opposite_neighbour = None

        for i in range(1,4):
            x_dist = math.fabs(centers[A][0]-centers[i][0])
            y_dist = math.fabs(centers[A][1]-centers[i][1])

            if x_dist < D_dist:
                D_dist = x_dist
                D = i
            if y_dist < B_dist:
                B_dist = y_dist
                B = i

        if B == D:
            continue

        C = [i for i in range(1,4) if i not in [B,D]][0]

        # "fix" the x-axis
        y_delta = math.fabs((centers[A][1]-centers[D][1]) - (centers[B][1]-centers[C][1]))
        x_delta = math.fabs((centers[A][0]-centers[B][0]) - (centers[D][0]-centers[C][0]))

        if max(y_delta,x_delta) < 10:
            # x,y = zip(*[centers[i] for i in [A,B,C,D]])
            # x = list(x)
            # y = list(y)
            # x.append(x[0])
            # y.append(y[0])
            # print len(x),len(y)
            # plt.plot(x,y,"o-",color="red")
            good_parallelograms.append(subset)
            ds.append(max(y_delta,x_delta))

        m.append(max(x_delta,y_delta))

        # assert len(opposite_neighbour) == 1
        # x_dist = math.fabs(centers[i_1][0]-centers[opposite_neighbour[0]][0])
        # y_dist = math.fabs(centers[i_1][1]-centers[opposite_neighbour[0]][1])
        # print x_dist,y_dist
    # print ds
    return good_parallelograms

def get_alg_pts(subject_id):
    alg_pts = []
    f_name = jungle.__image_setup__(subject_id)
    # print subject_id

    img = cv2.imread(f_name)

    img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    # ax1.imshow(close)
    #
    # # print close[70][559]
    # plt.show()


    # for c in close:
    #     print min(c)
    ix = np.in1d(close.ravel(),range(160)).reshape(close.shape)
    y,x = np.where(ix)

    # fig, ax1 = plt.subplots(1, 1)
    # ax1.imshow(close)
    # plt.plot(x,y,".",color="blue")
    # plt.ylim((close.shape[0],0))
    # plt.xlim((0,close.shape[1]))
    # plt.show()

    X = np.asarray(zip(x,y))
    db = DBSCAN(eps=5, min_samples=3).fit(X)

    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print n_clusters_
    unique_labels = set(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask = (labels == k)

        pts = X[class_member_mask]
        alg_pts.append(pts)


    if len(alg_pts) != 4:
        fig, [ax1,ax2] = plt.subplots(1, 2)
        ax1.imshow(img)
        # plt.show()

        # fig, ax1 = plt.subplots(1, 1)

        ax2.imshow(close)
        plt.ylim((close.shape[0],0))
        plt.xlim((0,close.shape[1]))
        # plt.show()
        plt.savefig("/home/ggdhines/bad_jungle_"+str(subject_id)+".png",bbox_inches='tight', pad_inches=0)
        plt.close()

    if len(alg_pts) >= 10:
        # print "too many"
        return []
    elif len(alg_pts) > 4:
        # print "correcting"
        good_pts = correction(f_name,alg_pts)
        if len(good_pts) == 1:
            return good_pts[0]
        else:
            return []
    else:
        return alg_pts

jungle = AggregationAPI(153,"development")
jungle.__setup__()

postgres_cursor = jungle.postgres_session.cursor()
postgres_cursor.execute("select subject_ids,annotations,user_id from classifications where project_id = 153")

markings = {}

markings_to_users = {}

for subject_ids,annotations,user_id in postgres_cursor.fetchall():
    if subject_ids == []:
        continue
    s = subject_ids[0]
    for task in annotations:
        if task["task"] == "T2":
            try:
                m = task["value"][0]["points"]

                if s not in markings:
                    markings[s] = []
                for i in m:
                    x = i["x"] + random.uniform(0,0.0001)
                    y = i["y"] + random.uniform(0,0.0001)

                    if s in markings:
                        assert (x,y) not in markings[s]
                    markings[s].append((x,y))
                    markings_to_users[(x,y)] = user_id

                # if s not in markings:
                #     markings[s] = [(i["x"],i["y"]) for i in m]
                # else:
                #     markings[s].extend([(i["x"],i["y"]) for i in m])
            except (KeyError,IndexError) as e:
                pass

accuracy = {}
alg_accuracy = []

problem_subjects = 0
empty_subjects = 0

for subject_id,m__ in markings.items():
    # fig, ax1 = plt.subplots(1, 1)
    # f_name = jungle.__image_setup__(subject_id)
    # img = cv2.imread(f_name)
    # ax1.imshow(img)
    user_pts = get_user_pts(m__)
    # plt.ylim((img.shape[0],0))
    # plt.xlim((0,img.shape[1]))
    # plt.savefig("/home/ggdhines/users_jungle_"+str(subject_id)+".png",bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.close()
    # continue

    try:
        alg_pts = get_alg_pts(subject_id)
    except KeyError:
        continue

    if len(user_pts) < 4:
        print "skipping due to users"
        empty_subjects += 1
        continue

    if len(alg_pts) != 4:
        problem_subjects += 1
        print "error subject"
        continue

    # fig, ax1 = plt.subplots(1, 1)
    # f_name = jungle.__image_setup__(subject_id)
    # img = cv2.imread(f_name)
    # ax1.imshow(img)

    user_centers = []
    for cluster in user_pts:
        cluster_x,cluster_y = zip(*cluster)
        x = np.median(cluster_x)
        y = np.median(cluster_y)
        user_centers.append((np.median(cluster_x),np.median(cluster_y)))
        # plt.plot(x,y,"o",color="red")

    alg_centers = []
    for cluster in alg_pts:
        cluster_x,cluster_y = zip(*cluster)
        x = np.median(cluster_x)
        y = np.median(cluster_y)
        alg_centers.append((x,y))
        # plt.plot(x,y,"o",color="blue")

    # plt.show()

    user_to_alg = []
    for user_index,(x,y) in enumerate(user_centers):
        closest_alg = None
        closest_dist = float("inf")

        for alg_index,(x2,y2) in enumerate(alg_centers):
            dist = math.sqrt((x-x2)**2+(y-y2)**2)

            if dist < closest_dist:
                closest_dist = dist
                closest_alg = alg_index

        user_to_alg.append(closest_alg)

    alg_to_user = []
    for alg_index,(x2,y2) in enumerate(alg_centers):
        closest_user = None
        closest_dist = float("inf")

        for user_index,(x,y) in enumerate(user_centers):
            dist = math.sqrt((x-x2)**2+(y-y2)**2)

            if dist < closest_dist:
                closest_dist = dist
                closest_user = user_index

        alg_to_user.append(closest_user)

    error = False
    for alg_index,user_index in enumerate(alg_to_user):
        if user_to_alg[user_index] != alg_index:
            error = True
    if error:
        print "problem image"
        continue

    for alg_index,user_index in enumerate(alg_to_user):
        if user_to_alg[user_index] != alg_index:
            print user_to_alg
            print alg_to_user
        assert user_to_alg[user_index] == alg_index
        x,y = user_centers[user_index]
        x2,y2 = alg_centers[alg_index]

        dist = math.sqrt((x-x2)**2+(y-y2)**2)
        alg_accuracy.append(dist)

        if dist > 3:
            print "***********"
            print len(user_pts[user_index])

            fig, ax1 = plt.subplots(1, 1)
            f_name = jungle.__image_setup__(subject_id)
            img = cv2.imread(f_name)
            ax1.imshow(img)


            for alg_index,user_index in enumerate(alg_to_user):
                x_,y_ = zip(*user_pts[user_index])
                plt.plot(x_,y_,".",color="green")
                x_,y_ = alg_centers[alg_index]
                plt.plot(x_,y_,".",color="red")

            plt.ylim((img.shape[0],0))
            plt.xlim((0,img.shape[1]))
            plt.savefig("/home/ggdhines/_jungle_"+str(subject_id)+".png",bbox_inches='tight', pad_inches=0)
            plt.show()

            img = cv2.imread(f_name)

            img = cv2.GaussianBlur(img,(5,5),0)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
            close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)

            fig, ax1 = plt.subplots(1, 1)
            # f_name = jungle.__image_setup__(subject_id)
            # img = cv2.imread(f_name)
            ax1.imshow(close)
            # user_pts = get_user_pts(markings)
            plt.ylim((img.shape[0],0))
            plt.xlim((0,img.shape[1]))
            plt.savefig("/home/ggdhines/close_jungle_"+str(subject_id)+".png",bbox_inches='tight', pad_inches=0)
            plt.show()
            # plt.close()



        for x3,y3 in user_pts[user_index]:
            dist = math.sqrt((x-x3)**2+(y-y3)**2)
            user_id = markings_to_users[(x3,y3)]

            if user_id not in accuracy:
                accuracy[user_id] = [dist]
            else:
                accuracy[user_id].append(dist)

    print len(alg_accuracy)

print "aaaa"
print len(markings)
print problem_subjects
print empty_subjects
print len(alg_accuracy)
print np.mean(alg_accuracy),np.median(alg_accuracy),max(alg_accuracy)
print "==="
for a in accuracy.values():
    print np.mean(a),np.median(a),max(a),len(a)


