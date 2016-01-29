__author__ = 'ggdhines'
from penguin import Penguins
import numpy
import math
import scipy

project = Penguins()
subjects = project.__get_retired_subjects__(1,False)

jj = 0

for zooniverse_id in subjects:
    subject = project.subject_collection.find_one({"zooniverse_id":zooniverse_id})
    count = subject["classification_count"]

    raw_classifications,raw_markings = project.__sort_annotations__(-1,[zooniverse_id],False)
    # clusters = project.__get_users_per_cluster__(-1,zooniverse_id,1,"point")
    clustering_aggregations = project.__cluster__(raw_markings)

    print subject
    # print raw_markings
    centers = []
    for cluster_index,cluster in clustering_aggregations[zooniverse_id][1]["point clusters"].items():
        if not(cluster_index in ["param","all_users"]):
            # print cluster_index
            # print cluster
            centers.append((cluster["center"][0],cluster["center"][1],cluster["users"]))

    if centers == []:
        continue

    x,y,u = zip(*centers)
    width = max(max(x)-min(x),max(y)-min(y))/2.
    center = numpy.median(x),numpy.median(y)

    # t = QuadTree(center,width,centers)
    tree = scipy.spatial.KDTree(zip(x,y))
    for p in zip(x,y):
        print tree.query(p,k=2,distance_upper_bound=20)





    print
    jj += 1
    if jj == 10:
        break