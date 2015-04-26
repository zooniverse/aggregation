__author__ = 'ggdhines'
import clustering
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,dendrogram
import time
from numpy import array

class Agglomerative(clustering.Cluster):
    def __init__(self,project_api,min_cluster_size=1):
        clustering.Cluster.__init__(self,project_api,min_cluster_size)
        self.algorithm_name = "agglomerative"

    def levenshtein(self,s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1       # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]/float(len(s1))

    def __fit__(self,markings,user_ids,jpeg_file=None,debug=False):
        start = time.time()

        cluster_centers = []
        end_clusters = []
        end_users = []

        l = [[(u,m[0],m[1]) for m in marking] for u,marking in zip(user_ids,markings)]
        user_list,pts_list,string_list = zip(*[item for sublist in l for item in sublist])

        # for i in range(len(string_list)):
        #     for j in range(i+1,len(string_list)):
        #         if self.levenshtein(string_list[i],string_list[j]) > 25:
        #             # print i,j
        #             print string_list[i],string_list[j]

        def f(u,v):
            return self.levenshtein(string_list[int(u[0])],string_list[int(v[0])])

        w = array([[i,] for i in range(len(string_list))])
        # Y = pdist(w,f)

        labels = [str(i) for i in w]
        row_dist = pd.DataFrame(squareform(pdist(w, f)), columns=labels, index=labels)

        row_clusters = linkage(row_dist, method='ward')


        clusters = [[(ip,pt,s)] for ip,pt,s in zip(user_list,pts_list,string_list)]
        # dendrogram(row_clusters)
        string_clusters = [[s,] for s in string_list]
        # print row_clusters
        for c1,c2,dist,t in row_clusters:
            if dist < 0.65:
                # print dist
                # print clusters[int(c1)]
                # # for s in clusters[int(c1)]:
                # #     index = string_list.index(s)
                # #     print user_list[index]
                # print clusters[int(c2)]
                # # for s in clusters[int(c2)]:
                # #     index = string_list.index(s)
                # #     print user_list[index]
                # print

                t = clusters[int(c1)][:]
                t.extend(clusters[int(c2)])
                clusters.append(t)
            else:
                if (clusters[int(c1)] is not None) and (clusters[int(c2)] is not None):
                    users1,pts,strings1 = zip(*clusters[int(c1)])
                    users2,pts,strings2= zip(*clusters[int(c2)])

                if clusters[int(c1)] is not None:
                    users,pts,strings = zip(*clusters[int(c1)])
                    if not(len(users) == len(list(set(users)))):
                        print users
                        print strings
                        print pts
                    # assert len(users) == len(list(set(users)))
                    else:
                        cluster_centers.append([np.median(axis) for axis in zip(*pts)])
                        # print strings
                        end_clusters.append(pts)
                        end_users.append(users)
                        string_clusters.append(strings)

                if clusters[int(c2)] is not None:
                    users,pts,strings = zip(*clusters[int(c2)])
                    if not(len(users) == len(list(set(users)))):
                        print users
                        print strings
                        print pts
                    # assert len(users) == len(list(set(users)))
                    # print strings
                    else:
                        cluster_centers.append([np.median(axis) for axis in zip(*pts)])
                        end_clusters.append(pts)
                        end_users.append(users)
                        string_clusters.append(strings)

                clusters.append(None)
        end = time.time()
        # for s in string_clusters:
        #     print len(s)
        #     print s
        # assert False
        return (cluster_centers, end_clusters,end_users),end-start