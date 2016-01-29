__author__ = 'greg'

def find_cluster(p,c_list):
    for i,c in enumerate(c_list):
        if p in c:
            return i

    return -1

def cluster_compare(c_list1,c_list2):
    #return those points in c_list2 which do not have a corresponding cluster in c_list1
    mapping_1_to_2 = [None for c in c_list1]
    #mapping_2_to_1 = [set() for c in c_list2]

    #print c_list1
    #print c_list2

    found = set()

    for i,c1 in enumerate(c_list1):
        #find which clusters each of those points wound up in
        mapping_1_to_2[i] = set([find_cluster(p,c_list2) for p in c1 if find_cluster(p,c_list2) >= 0])
        found = found.union(mapping_1_to_2[i])

    #print mapping_1_to_2
    not_found = [i for i in range(len(c_list2)) if not(i in found)]
    return not_found


def cluster_intersection_size(user_clusters,gold_clusters):
    missing_clusters = cluster_compare(user_clusters,gold_clusters)
    intersection_size = len(gold_clusters) - len(missing_clusters)

    return intersection_size


def metric(user_clusters,gold_clusters):
    intersection_size = cluster_intersection_size(user_clusters,gold_clusters)

    if (len(user_clusters) == 0) and (len(gold_clusters) == 0):
        return 1
    elif (len(user_clusters) == 0) or (len(gold_clusters) == 0):
        return 0
    else:
        return min(intersection_size/float(len(user_clusters)),intersection_size/float(len(gold_clusters)))

def metric2(user_clusters,gold_clusters):
    intersection_size = cluster_intersection_size(user_clusters,gold_clusters)

    if (len(user_clusters) == 0) and (len(gold_clusters) == 0):
        return 1
    elif (len(user_clusters) == 0):
        return 0.25*(intersection_size/float(len(gold_clusters))) + 0.75
    elif (len(gold_clusters) == 0):
        return 0.25 + 0.75*intersection_size/float(len(user_clusters))
    else:
        return 0.75*(intersection_size/float(len(user_clusters))) + 0.25*(intersection_size/float(len(gold_clusters)))