import math
import numpy as np

class MultiClickCorrect:
    def __init__(self,dist_threshold=float("inf"),overlap_threshold=float("inf"),min_cluster_size=0):
        self.dist_threshold = dist_threshold
        self.overlap_threshold = overlap_threshold
        self.min_cluster_size = min_cluster_size

    def __find_closest__(self, cluster_centers,users_per_cluster):
        # note this property is not symmetric - just because A is closest B, does not mean that B is closest to A
        # this means that for each cluster, we must read through ALL of the other cluster
        # however, this does mean that we will have some doubles which will need to be filtered out -
        # hence, closest_neighbours is a set
        # find the closest neighbours which satisfy an optional set of constraints
        closest_neighbours = set([])
        if len(cluster_centers) == 1:
            return []

        for c1_index,c1 in enumerate(cluster_centers):
            closest = None
            min_dist = float("inf")
            current_overlap = None


            for c2_index,c2 in enumerate(cluster_centers):
                if c1_index == c2_index:
                    continue

                dist = math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)
                users_1 = users_per_cluster[c1_index]
                users_2 = users_per_cluster[c2_index]
                overlap = [u for u in users_1 if u in users_2]

                if min_dist>dist:
                    min_dist = dist
                    closest = c2_index
                    current_overlap = overlap[:]

            min_size = min(len(users_per_cluster[c1_index]),len(users_per_cluster[c2_index]))
            if (len(current_overlap) <= self.overlap_threshold) and (min_dist <= self.dist_threshold) and (min_size >= self.min_cluster_size):
                closest_neighbours.add((min(c1_index,closest),max(c1_index,closest),tuple(current_overlap)))

        return list(closest_neighbours)

    def __fix__(self,cluster_results):
        cluster_centers,cluster_pts,users_per_cluster = cluster_results
        # find nearby clusters which have no users in common and merge them
        # mostly just to be on the results of dk-means
        while True:
            # compare every pair of clusters - returns only those clusters with 0 users in common
            # within the threshold
            closest_neighbours = self.__find_closest__(cluster_centers,users_per_cluster)
            if closest_neighbours == []:
                break

            # do this one at a time just to be careful
            c1_index, c2_index, overlap = closest_neighbours[0]

            # do this in the right order
            if c2_index > c1_index:
                jointClusters = cluster_pts.pop(c2_index)
                jointClusters.extend(cluster_pts.pop(c1_index))

                jointUsers = users_per_cluster.pop(c2_index)
                jointUsers.extend(users_per_cluster.pop(c1_index))

                cluster_centers.pop(c2_index)
                cluster_centers.pop(c1_index)
            else:
                jointClusters = cluster_pts.pop(c1_index)
                jointClusters.extend(cluster_pts.pop(c2_index))

                jointUsers = users_per_cluster.pop(c1_index)
                jointUsers.extend(users_per_cluster.pop(c2_index))

                cluster_centers.pop(c1_index)
                cluster_centers.pop(c2_index)

            X,Y = zip(*jointClusters)
            c = (np.mean(X),np.mean(Y))

            cluster_centers.append(c[:])
            cluster_pts.append(jointClusters[:])
            users_per_cluster.append(jointUsers[:])

        return cluster_centers,cluster_pts,users_per_cluster