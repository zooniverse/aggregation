__author__ = 'ggdhines'
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import six
from matplotlib import colors
import math

class DivisiveKmeans:
    def __init__(self, min_samples=1):
        self.min_samples = min_samples




    # def fit(self, markings,user_ids,jpeg_file=None,debug=False):
    #     clusters_to_go = []
    #     clusters_to_go.append((markings,user_ids,1))
    #
    #     print user_ids
    #
    #     end_clusters = []
    #     cluster_centers = []
    #
    #     while True:
    #         #if we have run out of clusters to process, break (hopefully done :) )
    #         if clusters_to_go == []:
    #             break
    #         m_,u_,num_clusters = clusters_to_go.pop(0)
    #
    #         #increment by 1
    #         kmeans = KMeans(init='k-means++', n_clusters=num_clusters+1, n_init=10).fit(markings)
    #
    #         labels = kmeans.labels_
    #         unique_labels = set(labels)
    #         for k in unique_labels:
    #             users = [ip for index,ip in enumerate(u_) if labels[index] == k]
    #             points = [pt for index,pt in enumerate(m_) if labels[index] == k]
    #
    #             #if the cluster does not have the minimum number of points, just skip it
    #             if len(points) < self.min_samples:
    #                 continue
    #
    #             #we have found a "clean" - final - cluster
    #             if len(set(users)) == len(users):
    #                 end_clusters.append(points)
    #                 X,Y = zip(*points)
    #                 cluster_centers.append((np.mean(X),np.mean(Y)))
    #             else:
    #                 clusters_to_go.append((points,users,num_clusters+1))
    #
    #
    #
    #
    #     if debug:
    #         return cluster_centers, end_clusters,total_noise2
    #     else:
    #         return cluster_centers

    def __fit__(self, markings,user_ids,jpeg_file=None,debug=False):
        #check to see if we need to split at all, i.e. there might only be one animal in total

        total = 0

        if len(user_ids) == len(list(set(user_ids))):
            #do these points meet the minimum threshold value?
            if len(markings) >= self.min_samples:
                X,Y = zip(*markings)
                cluster_centers = [(np.mean(X),np.mean(Y)), ]
                end_clusters = [markings,]
                if True:
                    return cluster_centers, end_clusters,user_ids
                else:
                    return cluster_centers
            else:
                if debug:
                    return [], [],[]
                else:
                    return []

        clusters_to_go = []
        clusters_to_go.append((markings,user_ids,1))

        end_clusters = []
        cluster_centers = []
        end_users = []

        colors_ = list(six.iteritems(colors.cnames))

        while True:
            # if we have run out of clusters to process, break (hopefully done :) )
            if clusters_to_go == []:
                break
            m_,u_,num_clusters = clusters_to_go.pop(-1)

            if jpeg_file is not None:
                image_file = cbook.get_sample_data(jpeg_file)
                image = plt.imread(image_file)
                fig, ax = plt.subplots()
                im = ax.imshow(image)

                X,Y = zip(*markings)
                #X = [1.875 *x for x in X]
                #Y = [1.875 *y for y in Y]
                plt.plot(X,Y,'.',color="blue")

                X,Y = zip(*m_)
                #X = [1.875 *x for x in X]
                #Y = [1.875 *y for y in Y]
                plt.plot(X,Y,'.',color="red")
                plt.show()

            #increment by 1
            while True:
                num_clusters += 1
                try:
                    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10).fit(m_)
                    total += 1
                except ValueError:
                    #all of these are noise - since we can't actually separate them
                    break

                labels = kmeans.labels_
                unique_labels = set(labels)
                temp_end_clusters = []
                temp_clusters_to_go = []
                temp_cluster_centers = []
                temp_users = []

                for k in unique_labels:
                    users = [ip for index,ip in enumerate(u_) if labels[index] == k]
                    points = [pt for index,pt in enumerate(m_) if labels[index] == k]
                    assert(users != [])

                    #if the cluster does not have the minimum number of points, just skip it
                    if len(points) < self.min_samples:
                        continue

                    #we have found a "clean" - final - cluster
                    if len(set(users)) == len(users):
                        temp_end_clusters.append(points)
                        X,Y = zip(*points)
                        temp_cluster_centers.append((np.mean(X),np.mean(Y)))
                        temp_users.append(users)
                    else:
                        temp_clusters_to_go.append((points,users,1))

                if temp_end_clusters != []:
                    end_clusters.extend(temp_end_clusters)
                    cluster_centers.extend(temp_cluster_centers)
                    clusters_to_go.extend(temp_clusters_to_go)
                    end_users.extend(temp_users)
                    break

        #print total

        for c in end_clusters:
            assert(len(c) >= self.min_samples)

        return cluster_centers, end_clusters,end_users
