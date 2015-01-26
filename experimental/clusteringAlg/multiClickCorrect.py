__author__ = 'greg'
import math
import os
import sys

# add the paths necessary for clustering algorithm and ibcc - currently only works on Greg's computer
if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
elif os.path.exists("/Users/greg"):
    sys.path.append("/Users/greg/Code/reduction/experimental/clusteringAlg")
    sys.path.append("/Users/greg/Code/pyIBCC/python")
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"

else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
    sys.path.append("/home/greg/github/pyIBCC/python")
import ibcc

class Fix:
    def __init__(self):
        pass

    def calc_relations(self,centers,clusters,pts,user_list,dist_threshold=float("inf"),user_threshold=float("inf")):
        relations = []
        for c1_index in range(len(clusters)):
            for c2_index in range(c1_index+1,len(clusters)):
                c1 = centers[c1_index]
                c2 = centers[c2_index]

                dist = math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)
                users_1 = [user_list[pts.index(pt)] for pt in clusters[c1_index]]
                users_2 = [user_list[pts.index(pt)] for pt in clusters[c2_index]]

                overlap = [u for u in users_1 if u in users_2]
                #print (len(overlap),dist)

                #print (len(overlap),dist)
                if (len(overlap) <= user_threshold) and (dist <= dist_threshold):
                    relations.append((dist,overlap,c1_index,c2_index))

        relations.sort(key= lambda x:x[0])
        relations.sort(key= lambda x:len(x[1]))

        return relations

    def __find_closest__(self,centers,clusters,users_per_cluster,dist_threshold=float("inf"),user_threshold=float("inf")):
        # find the closest neighbours which satisfy an optional set of constraints
        closest_neighbours = []
        for c1_index in range(len(clusters)):
            closest = None
            min_dist = float("inf")
            overlap_size = None


            for c2_index in range(len(clusters)):
                if c1_index == c2_index:
                    continue

                c1 = centers[c1_index]
                c2 = centers[c2_index]

                dist = math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)
                #users_1 = [user_list[pts.index(pt)] for pt in clusters[c1_index]]
                #users_2 = [user_list[pts.index(pt)] for pt in clusters[c2_index]]
                users_1 = users_per_cluster[c1_index]
                users_2 = users_per_cluster[c2_index]
                overlap = len([u for u in users_1 if u in users_2])

                if min_dist>dist:
                    min_dist = dist
                    closest = c2_index
                    overlap_size = overlap

            if (overlap_size <= user_threshold) and (min_dist <= dist_threshold):
                closest_neighbours.append((c1_index,closest))

        return closest_neighbours




    def __fix__(self,results):#centers,clusters,users_per_cluster):#,users_per_cluster,threshold,f_name=None):
        #look for abnormally close clusters with only one or zero users in common
        #zero is possible as a result of how k-means splits points
        #users per cluster must in the same order as the points per cluster
        while True:
            #compare every pair of clusters - returns only those clusters with 0 or 1 users in common
            #within the threshold
            relations = self.calc_relations(centers,users_per_cluster,threshold)
            if relations == []:
                break

            #look at the closest pair
            overlap = relations[0][1]
            distance = relations[0][0]
            c1_index = relations[0][2]
            c2_index = relations[0][3]

            #make sure to pop in the right order so else the indices will get messed up
            if c2_index > c1_index:
                cluster_2 = clusters.pop(c2_index)
                cluster_1 = clusters.pop(c1_index)

                cent_2 = centers.pop(c2_index)
                cent_1 = centers.pop(c1_index)

                users_2 = users_per_cluster.pop(c2_index)
                users_1 = users_per_cluster.pop(c1_index)

            else:
                cluster_1 = clusters.pop(c1_index)
                cluster_2 = clusters.pop(c2_index)

                cent_1 = centers.pop(c1_index)
                cent_2 = centers.pop(c2_index)

                users_1 = users_per_cluster.pop(c1_index)
                users_2 = users_per_cluster.pop(c2_index)

            #the distance < threshold check may be redundant - since calc_relations can also check
            #but doesn't hurt
            if (overlap != []) and (distance < threshold):
                assert(len(overlap) == 1)
                overlap_user = overlap[0]
                #the point in each cluster which corresponds to the overlapping user
                p1 = cluster_1.pop(users_1.index(overlap_user))
                #double check that there no other equaivalent points in the array
                #shouldn't happen - unless by really bad luck
                assert sum([1 for p in cluster_1 if (p == p1)]) == 0

                p2 = cluster_2.pop(users_2.index(overlap_user))
                #double check that there no other equaivalent points in the array
                #shouldn't happen - unless by really bad luck
                assert sum([1 for p in cluster_1 if (p == p2)]) == 0

                #since we have removed the points, we now remove the users themselves
                users_1.remove(overlap_user)
                users_2.remove(overlap_user)

                #now merge
                #for now, only deal with 2D points - which should always be the case
                assert len(p1) == 2
                avg_pt = ((p1[0]+p2[0])/2., (p1[1]+p2[1])/2.)
                joint_cluster = cluster_1[:]
                joint_cluster.extend(cluster_2)
                joint_cluster.append(avg_pt)

                joint_users = users_1[:]
                joint_users.extend(users_2)
                joint_users.append(overlap_user)

                #recalculate the center
                X,Y = zip(*joint_cluster)
                center = (np.mean(X),np.mean(Y))

                #add this new merged cluster back
                centers.append(center)
                clusters.append(joint_cluster)
                users_per_cluster.append(joint_users)

            else:
                assert overlap == []
                #we have no nearby clusters with no common users - should be easy to merge

                joint_cluster = cluster_1[:]
                joint_cluster.extend(cluster_2)

                joint_users = users_1[:]
                joint_users.extend(users_2)

                #recalculate the center
                X,Y = zip(*joint_cluster)
                center = (np.mean(X),np.mean(Y))

                #add this new merged cluster back
                centers.append(center)
                clusters.append(joint_cluster)
                users_per_cluster.append(joint_users)

        return centers,clusters,users_per_cluster

    def __ibcc__(self,results):
        with open(base_directory+"/Databases/fix_ibcc.py","wb") as f:
            f.write("import numpy as np\n")
            f.write("scores = np.array([0,1])\n")
            f.write("nScores = len(scores)\n")
            f.write("nClasses = 2\n")
            f.write("inputFile = \""+base_directory+"/Databases/fix_ibcc.csv\"\n")
            f.write("outputFile = \""+base_directory+"/Databases/fix_ibcc.out\"\n")
            f.write("confMatFile = \""+base_directory+"/Databases/fix_ibcc.mat\"\n")
            f.write("nu0 = np.array([30,70])\n")
            f.write("alpha0 = np.array([[3, 1], [1,3]])\n")

        with open(base_directory+"/Databases/fix_ibcc.csv","wb") as f:
            f.write("a,b,c\n")
