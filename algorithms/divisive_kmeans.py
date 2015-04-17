__author__ = 'ggdhines'
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import six
from matplotlib import colors
import math
import time
from clustering import Cluster


class DivisiveKMeans(Cluster):

    def __init__(self, project_api,min_cluster_size=1,fix_distinct_clusters=False):
        Cluster.__init__(self,project_api,min_cluster_size)

        self.fix_distinct_clusters = fix_distinct_clusters
        self.algorithm_name = "divisivekmeans"

    def __cluster_subject__(self,subject_id,max_users=None,jpeg_file=None):
        """
        override the parent method but still call it - also for correcting problems with nearest neighbours etc.
        things that only make sense with divisive k-means
        :param subject_id:
        :param jpeg_file:
        :return:
        """
        time_to_cluster = Cluster.__cluster_subject__(self,subject_id,max_users,jpeg_file)

        # kmeans will sometimes split clusters such that we have two nearest neighbour clusters with no users
        # in common - both representing the same animal/thing. Do we want to find such cases and fix them?
        # in this case fixing is just a case of merging the clusters
        # if self.fix_distinct_clusters:
        #     start = time.time()
        #     self.clusterResults[subject_id] = self.correction.__fix__(self.clusterResults[subject_id])
        #     end = time.time()
        #
        #     return time_to_cluster + (end-start)
        # else:
        return time_to_cluster

    def __correct__(self,subject_id,debug=False):
        """
        find any nearest neighbour tuples of clusters which have no users in common and merge them
        :return:
        """
        results = self.clusterResults[subject_id]
        i = 0
        # the length of results[2] may and probably will change as we correct things
        # so don't use a for loop
        # -1 so we always have at least one more element to compare against
        while i < len(results[2])-1:
            users_i = results[2][i]
            pts_i = results[1][i]
            cluster_i = results[0][i]

            closest_distance = float("inf")
            closest_neighbour = None
            overlap = None
            # check the overlap between i and all clusters "above" it - overlap is symmetrical so we don't need
            # to check both ways. Also we are going backwards so that we can pop stuff from the list without
            # messing the indexing up
            for j in range(len(results[2])-1,i,-1):
                assert j != i
                users_j = results[2][j]
                cluster_j = results[0][j]
                dist = math.sqrt(sum([(pi-pj)**2 for (pi,pj) in zip(cluster_i,cluster_j)]))

                if dist < closest_distance:
                    closest_distance = dist
                    overlap = [u for u in users_j if u in users_i]
                    closest_neighbour = j

            if (len(overlap) == 0) and (closest_distance < 10):
                if debug:
                    x = [results[0][i][0],results[0][j][0]]
                    y = [results[0][i][1],results[0][j][1]]

                    ax = self.project_api.__display_image__(subject_id,[[x,y]],[{"color":"red"}])

                # remove the j'th element and merge it with the i'th one
                center = results[0].pop(closest_neighbour)
                pts = results[1].pop(closest_neighbour)
                users = results[2].pop(closest_neighbour)

                # to allow for generalizations where the overlap is non-empty, we need  a way to merge points
                for users in overlap:
                    # todo: do generalization
                    pass

                results[1][i].extend(pts)
                results[2][i].extend(users)

                # calculate the new center
                results[0][i] = [np.mean(axis) for axis in zip(*results[1][i])]
            # move on to the next element
            i += 1

        # print "- " +str(len(self.clusterResults[subject_id][0]))

    def __fit__(self,markings,user_ids,jpeg_file=None,debug=False):
        """
        the main function - currently works for any number of dimensions
        :param markings: the actual markings
        :param user_ids: who has marked this image
        :param jpeg_file: in case we need to show the step by step of how this algorithm works - for debugging
        :return:
        """
        # associate each marking with their correspondng user and extract only the relevant part of the marking for
        # the clustering

        l = [[(u,m[0]) for m in marking] for u,marking in zip(user_ids,markings)]
        user_list,pts_list = zip(*[item for sublist in l for item in sublist])

        total_kmeans = 0
        start = time.time()
        # check to see if we need to split at all, i.e. there might only be one animal in total
        # or just one user has marked this image
        if len(user_list) == len(list(set(user_list))):
            # do these points meet the minimum threshold value?
            if len(pts_list) >= self.min_cluster_size:
                X,Y = zip(*pts_list)
                cluster_centers = [(np.mean(X),np.mean(Y)), ]
                end_clusters = [pts_list,]
                end = time.time()

                return (cluster_centers, end_clusters,[user_ids,]), end - start
            else:
                end = time.time()
                return ([],[], []), end - start

        # clusters_to_go is a stack where we push clusters onto which we still have to process
        # i.e. split into subclusters such that each cluster contains at most one marking from each user
        clusters_to_go = [(pts_list,user_list)]

        # this is what we return
        end_clusters = []
        cluster_centers = []
        end_users = []

        while True:
            # if we have run out of clusters to process, break (hopefully done :) )
            if clusters_to_go == []:
                break

            # get the next cluster off the stack
            m_,u_ = clusters_to_go.pop(-1)

            # if we are debugging - show the image
            # red markings are the ones currently being considered
            if jpeg_file is not None:
                image_file = cbook.get_sample_data(jpeg_file)
                image = plt.imread(image_file)
                fig, ax = plt.subplots()
                im = ax.imshow(image)

                X,Y = zip(*markings)
                plt.plot(X,Y,'.',color="blue")

                X,Y = zip(*m_)
                plt.plot(X,Y,'.',color="red")
                plt.show()

            # how many clusters to split into
            num_clusters = 0
            while True:
                num_clusters += 1

                # use kmeans to split into given number of clusters
                try:
                    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10).fit(m_)
                    total_kmeans += 1
                except ValueError:
                    # all of these are noise - since we can't actually separate them
                    # todo: check and see how often this happens and if so, if it should
                    break

                # now search through the resulting sub clusters and find any/all that have at most one
                # marking from each user - if so, these clusters are fixed and we can remove them
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

                    # if the cluster does not have the minimum number of points, just skip it
                    if len(points) < self.min_cluster_size:
                        continue

                    # quick way of checking if all of the markings are from different users
                    if len(set(users)) == len(users):
                        temp_end_clusters.append(points)

                        # find the center of this cluster - just take the average in every direction
                        # slightly complicated so that we don't presume how many dimensions we have
                        # todo: make having the median an option
                        temp_cluster_centers.append([np.median(axis) for axis in zip(*points)])
                        temp_users.append(users)
                    else:
                        temp_clusters_to_go.append((points,users))

                # if we have found one end/leaf subcluster, then we don't need to split this cluster up anymore
                # so stop and move onto the next one
                if temp_end_clusters != []:
                    end_clusters.extend(temp_end_clusters)
                    cluster_centers.extend(temp_cluster_centers)
                    clusters_to_go.extend(temp_clusters_to_go)
                    end_users.extend(temp_users)
                    break

        # sanity check to make sure that all clusters are the correct size
        for c in end_clusters:
            assert(len(c) >= self.min_cluster_size)
        end = time.time()
        # print "- " +str(len(cluster_centers))
        return (cluster_centers, end_clusters,end_users),end-start
