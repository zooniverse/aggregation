__author__ = 'ggdhines'
import clustering
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
import time
import automatic_optics

class Agglomerative(clustering.Cluster):
    def __init__(self,project_api,min_cluster_size=1):
        clustering.Cluster.__init__(self,project_api,min_cluster_size)
        self.algorithm_name = "agglomerative"

    def __add_cluster(self,cluster_centers,end_clusters,end_users,node):
        cluster_centers.append([np.mean(axis) for axis in zip(*node.pts)])
        end_clusters.append(node.pts)
        end_users.append(node.users)

        return cluster_centers,end_clusters,end_users

    def __fit__(self,markings,user_ids,jpeg_file=None,debug=False):
        start = time.time()

        cluster_centers = []
        end_clusters = []
        end_users = []

        l = [[(u,m[0]) for m in marking] for u,marking in zip(user_ids,markings)]
        user_list,pts_list = zip(*[item for sublist in l for item in sublist])

        labels = [str(i) for i in pts_list]
        df = pd.DataFrame(np.array(pts_list), columns=["X","Y"], index=labels)
        row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)

        row_clusters = linkage(row_dist, method='ward')
        nodes = [automatic_optics.LeafNode(pt,ii,user=user) for ii,(user,pt) in enumerate(zip(user_list,pts_list))]

        max_height = 0

        for merge in row_clusters:
            rchild_index = int(merge[0])
            lchild_index = int(merge[1])

            rnode = nodes[rchild_index]
            lnode = nodes[lchild_index]

            if (rnode is None) or (lnode is None):
                if rnode is not None:
                    cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,rnode)
                elif lnode is not None:
                    cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,lnode)
                nodes.append(None)
            else:
                cur_height = max(rnode.height,lnode.height)
                #print max_height,cur_height
                if False:#cur_height < (max_height-1):
                    cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,rnode)
                    cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,lnode)
                    nodes.append(None)
                else:
                    max_height = max(max_height,cur_height)


                    intersection = [u for u in rnode.users if u in lnode.users]
                    if intersection != []:
                        cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,rnode)
                        cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,lnode)

                        nodes.append(None)
                    else:
                        nodes.append(automatic_optics.InnerNode(rnode,lnode))

        end = time.time()
        print "- " +str(len(cluster_centers))
        return (cluster_centers, end_clusters,end_users),end-start