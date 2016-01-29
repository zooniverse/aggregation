__author__ = 'greg'
import math
import os
import sys
import numpy as np

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

def __ibcc__(results_dict,users_per_subject):
    # create a global index of all users
    global_users = []
    for u_list in users_per_subject.values():
        for u in u_list:
            if not(u in global_users):
                global_users.append(u)

    things_in_subject = {}
    things_list = []

    thing_index = 0
    off_by_one_clusters = []

    for zooinverse_id in results_dict:
        things_in_subject[zooinverse_id] = []

        centers,clusters,users = results_dict[zooinverse_id]
        pairs = __find_closest__(centers,clusters,users,user_threshold=1,offset=thing_index)
        off_by_one_clusters.extend(list(pairs))

        for users_per_marking in users:
            things_in_subject[zooinverse_id].append(thing_index)

            # find out who saw or did not see this "thing" - out of everyone who viewed this subject
            t = []
            for u in users_per_subject[zooinverse_id]:
                if u in users_per_marking:
                    t.append((global_users.index(u),1))
                else:
                    t.append((global_users.index(u),0))

            things_list.append(t[:])
            thing_index += 1

    # run ibcc with out combining any of the clusters
    with open(base_directory+"/Databases/base_ibcc.csv","wb") as f:
        f.write("a,b,c\n")
        for thing_index in range(len(things_list)):
            for user_index, marked in things_list[thing_index]:
                f.write(str(user_index)+","+str(thing_index)+","+str(marked)+"\n")
    __ibcc_init__("base")
    ibcc.runIbcc(base_directory+"/Databases/base_ibcc.py")

    # now try merging each possible pair and running ibcc on the resulting set up
    # yes, this is going to be tedious and time consuming - hope for a better implementation later on
    for count,(c1,c2,overlap) in enumerate(off_by_one_clusters):
        # most of the time, thing_index and thing_prime_index will be the same
        # but can be an off by one indifference to account for that fact that we are skipping over c2
        thing_prime_index = 0

        print (c1,c2)

        with open(base_directory+"/Databases/merged_ibcc.csv","wb") as f:
            f.write("a,b,c\n")
            for thing_index in range(len(things_list)):
                #print (thing_index,thing_prime_index)
                if thing_index == c2:
                    # we skipping this one
                    #print "skip"
                    pass
                else:
                    if thing_index == c1:
                        # merge
                        assert thing_index == thing_prime_index
                        assert len(list(overlap)) <= 1
                        #print zip(*things_list[c1])[0]
                        #print zip(*things_list[c1])[1]
                        #print zip(*things_list[c2])[1]
                        for (user_index1,marked1),(user_index2,marked2) in zip(things_list[c1],things_list[c2]):
                            assert user_index1 == user_index2
                            f.write(str(user_index1)+","+str(thing_prime_index)+","+str(marked1 or marked2)+"\n")
                    else:
                        # proceed as normal
                        # continue
                        for user_index, marked in things_list[thing_index]:
                            f.write(str(user_index)+","+str(thing_prime_index)+","+str(marked)+"\n")

                    thing_prime_index += 1

            #print thing_index
            #print thing_prime_index
            #assert thing_prime_index == (thing_index-1)

        __ibcc_init__("merged")
        ibcc.runIbcc(base_directory+"/Databases/merged_ibcc.py")
        p1 = load_ibcc_probabilities("base",c1)
        p2 = load_ibcc_probabilities("base",c2)
        p3 = load_ibcc_probabilities("merged",c1)
        print (p1,p2,p3)
        if p3 < (max(p1,p2)-0.01):
            break


def load_ibcc_probabilities(fname,subject_index):
    with open(base_directory+"/Databases/"+fname+"_ibcc.out","rb") as f:
        for lineIndex,line in enumerate(f.readlines()):
            if lineIndex == subject_index:
                return float(line[:-1].split(" ")[2])


def __ibcc_init__(fname):
    with open(base_directory+"/Databases/"+fname+"_ibcc.py","wb") as f:
        f.write("import numpy as np\n")
        f.write("scores = np.array([0,1])\n")
        f.write("nScores = len(scores)\n")
        f.write("nClasses = 2\n")
        f.write("inputFile = \""+base_directory+"/Databases/"+fname+"_ibcc.csv\"\n")
        f.write("outputFile = \""+base_directory+"/Databases/"+fname+"_ibcc.out\"\n")
        f.write("confMatFile = \""+base_directory+"/Databases/"+fname+"_ibcc.mat\"\n")
        f.write("nu0 = np.array([30,70])\n")
        f.write("alpha0 = np.array([[3, 1], [1,3]])\n")

    try:
        os.remove(base_directory+"/Databases/"+fname+".out")
    except OSError:
        pass

    try:
        os.remove(base_directory+"/Databases/"+fname+".mat")
    except OSError:
        pass

    try:
        os.remove(base_directory+"/Databases/"+fname+".csv.dat")
    except OSError:
        pass

def __confusion__():
    pass

def __ibcc__2(results_dict,users_per_subject):
    # results needs to be a dictionary which maps from a subject ID to a list of found clusters
    # users_per_subject needs to be different, just in case someone clicked on nothing
    # get a list of all of the users
    assert type(users_per_subject) == dict
    global_users = []
    for u_list in users_per_subject.values():
        for u in u_list:
            if not(u in global_users):
                global_users.append(u)

    things_in_subject = {}
    off_by_one_clusters = []
    things_list = []
    thing_index = 0

    for zooinverse_id in results_dict:
        things_in_subject[zooinverse_id] = []

        centers,clusters,users = results_dict[zooinverse_id]
        pairs = __find_closest__(centers,clusters,users,user_threshold=1,offset=thing_index)
        off_by_one_clusters.extend(list(pairs))

        for users_per_marking in users:
            things_in_subject[zooinverse_id].append(thing_index)

            # find out who saw or did not see this "thing" - out of everyone who viewed this subject
            t = []
            for u in users_per_subject[zooinverse_id]:
                if u in users_per_marking:
                    t.append((global_users.index(u),1))
                else:
                    t.append((global_users.index(u),0))

            things_list.append(t[:])
            thing_index += 1

    # run ibcc with out combining any of the clusters
    with open(base_directory+"/Databases/base_ibcc.csv","wb") as f:
        f.write("a,b,c\n")
        for thing_index in range(len(things_list)):
            for user_index, marked in things_list[thing_index]:
                f.write(str(user_index)+","+str(thing_index)+","+str(marked)+"\n")
    __ibcc_init__("base")
    ibcc.runIbcc(base_directory+"/Databases/base_ibcc.py")

    confusions = []
    with open(base_directory+"/Databases/base_ibcc.mat") as f:
        for user_index, l in enumerate(f.readlines()):
            confusions.append([float(f) for f in l[:-1].split(" ")])

    for count,(c1,c2,overlap) in enumerate(off_by_one_clusters):
        print things_list[c1]
        print things_list[c2]

        users = zip(*things_list[c1])[0]
        for u in users:
            print confusions[u][2],confusions[u][3]

        break
__author__ = 'greg'
