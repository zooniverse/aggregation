__author__ = 'ggdhines'
import clustering
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
import time
import automatic_optics
from scipy.stats import beta
import math
import numpy

class Agglomerative(clustering.Cluster):
    def __init__(self,project_api,min_cluster_size=1):
        clustering.Cluster.__init__(self,project_api,min_cluster_size)
        self.algorithm_name = "agglomerative"
        self.all_distances = []
        self.max = 0

    def __add_cluster(self,cluster_centers,end_clusters,end_users,node):
        # if len(node.pts) < 4:
        #     cluster_centers.append(None)
        #     end_clusters.append(None)
        #     end_users.append(None)
        # else:
        cluster_centers.append([np.median(axis) for axis in zip(*node.pts)])
        end_clusters.append(node.pts)
        end_users.append(node.users)

        return cluster_centers,end_clusters,end_users

    def __inner_fit__(self,markings,user_ids,jpeg_file=None,debug=False,gold_standard=False,subject_id=None):
        start = time.time()

        cluster_centers = []
        end_clusters = []
        end_users = []

        l = [[(u,m[0],m[1]) for m in marking] for u,marking in zip(user_ids,markings)]
        try:
            user_list,pts_list,text_list = zip(*[item for sublist in l for item in sublist])
        except ValueError:
            print markings
            print l
            raise
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

                # if False:#cur_height < (max_height-1):
                #     cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,rnode)
                #     cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,lnode)
                #     nodes.append(None)
                # else:
                #     max_height = max(max_height,cur_height)

                intersection = [u for u in rnode.users if u in lnode.users]
                pts1 = [rnode.pts[rnode.users.index(i)] for i in intersection]
                text1 = [text_list[pts_list.index(p)] for p in pts1]
                # print text1

                pts2 = [lnode.pts[lnode.users.index(i)] for i in intersection]
                text2 = [text_list[pts_list.index(p)] for p in pts2]
                # print text2
                # print [lnode.]

                # if text1 != []:
                #     pts1 = sorted(pts1,key = lambda x:x[0])
                #     pts2 = sorted(pts2,key = lambda x:x[0])
                #     print pts1
                #     print pts2
                #     assert pts1 != pts2
                #     print "===-0--"

                if intersection != []:
                    cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,rnode)
                    cluster_centers,end_clusters,end_users = self.__add_cluster(cluster_centers,end_clusters,end_users,lnode)

                    nodes.append(None)
                else:
                    nodes.append(automatic_optics.InnerNode(rnode,lnode))

        if nodes[-1] is not None:
            self.__add_cluster(cluster_centers,end_clusters,end_users,nodes[-1])

        end = time.time()
        # print "- " +str(len(cluster_centers))

        # if not gold_standard:
        #     for cluster_index in range(len(cluster_centers)):
        #
        #         center = cluster_centers[cluster_index]
        #         if center is None:
        #             continue
        #         cluster = end_clusters[cluster_index]
        #         assert len(cluster) >= 4
        #         print "==--"
        #         pts_and_dist = [(math.sqrt((center[0]-p[0])**2+(center[1]-p[1])**2),p) for p in cluster]
        #         # print pts_and_dist
        #         # self.all_distances.extend(distances)
        #         # self.max = max(self.max,max(distances))
        #         pts_and_dist.sort(key=lambda x:x[0])
        #         pts_and_dist = [(d[0]/(2.*pts_and_dist[-1][0]),d[1]) for d in pts_and_dist]
        #         distances,pts = zip(*pts_and_dist)
        #         assert len(distances) >= 4
        #         print len(distances)
        #         print range(2,len(distances)-1)
        #
        #
        #         for ii in range(2,len(distances)):
        #         # for ii in range(len(distances)-1,3,-1):
        #             # offset = -4
        #
        #             mean=numpy.mean(distances[:ii])
        #             var=numpy.var(distances[:ii],ddof=1)
        #
        #             # ii = len(distances)+offset
        #
        #             if var >= (mean*(1-mean)):
        #                 print "skipping " + str(ii)
        #                 continue
        #                 # ii -= 1
        #                 # mean=numpy.mean(distances[:ii])
        #                 # var=numpy.var(distances[:ii],ddof=1)
        #
        #             # print ii
        #             alpha1=mean*(mean*(1-mean)/var-1)
        #             beta1=alpha1*(1-mean)/mean
        #             # print ii,len(distances)
        #             # for d in sorted(distances):
        #             print ii, beta.cdf(distances[ii],alpha1,beta1)
        #             # if beta.cdf(distances[ii],alpha1,beta1) == 1.:
        #
        #         print
        #         args_l = [[[center[0],pts[-1][0]],[center[1],pts[-1][1]]]]
        #         self.project_api.__display_image__(subject_id,args_l,[{"color":"blue"}])
        #         # # assert var<mean*(1-mean)
        # # print self.max
        return (cluster_centers, end_clusters,end_users),end-start

    def __check__(self):
        self.all_distances = [d/max(self.all_distances) for d in self.all_distances]
        mean=numpy.mean(self.all_distances)
        var=numpy.var(self.all_distances,ddof=1)

        ii = len(self.all_distances)
        while var >= (mean*(1-mean)):
            ii -= 1
            mean=numpy.mean(self.all_distances[:ii])
            var=numpy.var(self.all_distances[:ii],ddof=1)

        alpha1=mean*(mean*(1-mean)/var-1)
        beta1=alpha1*(1-mean)/mean
        for d in sorted(self.all_distances):
            print d,beta.cdf(d,alpha1,beta1)