__author__ = 'greg'
from fix import Fix
import numpy as np
import sys
import os

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

class ZeroFix(Fix):
    def __init__(self):
        pass

    def __fix__(self,results):
        centers,clusters,users_per_cluster = results
        while True:
            # compare every pair of clusters - returns only those clusters with 0 users in common
            # within the threshold
            closest_neighbours = Fix.__find_closest__(self,centers,clusters,users_per_cluster,user_threshold=0)
            if closest_neighbours == []:
                break

            # do this one at a time just to be careful
            c1_index, c2_index = closest_neighbours[0]

            # do this in the right order
            if c2_index > c1_index:
                jointClusters = clusters.pop(c2_index)
                jointClusters.extend(clusters.pop(c1_index))

                jointUsers = users_per_cluster.pop(c2_index)
                jointUsers.extend(users_per_cluster.pop(c1_index))

                centers.pop(c2_index)
                centers.pop(c1_index)
            else:
                jointClusters = clusters.pop(c1_index)
                jointClusters.extend(clusters.pop(c2_index))

                jointUsers = users_per_cluster.pop(c1_index)
                jointUsers.extend(users_per_cluster.pop(c2_index))

                centers.pop(c1_index)
                centers.pop(c2_index)

            X,Y = zip(*jointClusters)
            c = (np.mean(X),np.mean(Y))

            centers.append(c[:])
            clusters.append(jointClusters[:])
            users_per_cluster.append(jointUsers[:])

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


