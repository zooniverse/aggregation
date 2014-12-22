__author__ = 'ggdhines'
import random
import math
from copy import deepcopy

class KMedoids:
    def __init__(self, min_samples):
        self.min_samples = min_samples

    def distance(self,(m1,u1),(m2,u2)):
        if m1 == m2:
            print (m1,u1)
            print (m2,u2)
            assert False
        elif u1 == u2:
            return float("inf")
        else:
            return math.sqrt((m1[0]-m2[0])**2+(m1[1]-m2[1])**2)

    def cost(self,cent,cluster):
        return sum([self.distance(pt,cent) for pt in cluster])

    def fit2(self, markings,user_ids,jpeg_file=None,debug=False):
        #print len(markings)

        clusters = []

        for n in range(2,len(markings)):
            centroids = random.sample(zip(markings,user_ids),n)
            clusters = [[(m,u)] for (m,u) in centroids]

            previous_clusters = None
            badOnes = []

            old_cost = None
            print "=== " + str(n)
            for iter_count in range(50):


                for m,u in zip(markings,user_ids):
                    if (m,u) in centroids:
                        continue

                    # if previous_clusters != None:
                    #     index = [True if (m,u) in c else False for c in previous_clusters].index(True)
                    #     cost = self.distance((m,u),clusters[index][0])
                    #     if (badOnes != []) and (index == badOnes[0]):
                    #         print cost

                    dist = [self.distance((m,u),cent) for cent in centroids]
                    cluster_index = dist.index(min(dist))

                    clusters[cluster_index].append((m,u))

                #reset the old centroids
                centroids = []

                total_cost = 0
                costs = []

                #print [len(c) for c in clusters]
                badOnes = []
                for c_index,c in enumerate(clusters):
                    #for each cluster, determine what the new centroid should be
                    min_cost = float("inf")
                    opt_centroid = None

                    for i in range(len(c)):
                        temp_centroid = c[i]
                        temp_cluster = c[:i]
                        temp_cluster.extend(c[i+1:])

                        #print i,len(temp_cluster),len(c)
                        assert not(temp_centroid in temp_cluster)

                        temp_cost = self.cost(temp_centroid,temp_cluster)
                        if temp_cost <= min_cost:
                            min_cost = temp_cost
                            opt_centroid = deepcopy(temp_centroid)

                    assert opt_centroid is not None
                    if len(c) == 1:
                        temp_cost = 0

                    total_cost += min_cost
                    centroids.append(opt_centroid)
                    costs.append(min_cost)

                    if temp_cost == float("inf"):
                        badOnes.append(c_index)

                if (old_cost is not None) and (old_cost == total_cost):
                    break

                old_cost = total_cost

                previous_clusters= deepcopy(clusters)
                clusters = [[cent,] for cent in centroids]
                #print total_cost


            clean = True
            for c in clusters:
                users = zip(*c)[1]
                if len(set(users)) != len(users):
                    clean = False
                    break

            if clean:
                print "^^^^ " + str(n)
                break
            # #need to check if all clusters are either "clean" or noise
            # clean = True
            # for k in unique_labels:
            #     users = [ip for index,ip in enumerate(user_ids) if labels[index] == k]
            #
            #     if len(users) < self.min_samples:
            #         continue
            #
            #     #we have found a "clean" - final - cluster
            #     if len(set(users)) != len(users):
            #         clean = False
            #         break
            #
            # if clean:
            #     break

        print n
        return None,None,None