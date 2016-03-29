#!/usr/bin/env python
from __future__ import print_function
import pymongo
import sys
sys.path.append("/home/ggdhines/github/aggregation/engine")
sys.path.append("/home/ggdhines/Pycharm/reduction/engine")
from agglomerative import Agglomerative
import csv

name_changes = {}
with open("/home/ggdhines/Downloads/Nomenclature_changes.csv","rb") as f:
    f.readline()
    reader = csv.reader(f,delimiter=",")

    for zoo_id,pre_zoo_id in reader:
        print(pre_zoo_id+"|")
        print(pre_zoo_id[0])
        if pre_zoo_id != "":
            name_changes[zoo_id] = pre_zoo_id[:-1]

assert False
roi_dict = {}

# method for checking if a given marking is within the ROI
def __in_roi__(self,site,marking):
    """
    does the actual checking
    :param object_id:
    :param marking:
    :return:
    """

    if site not in roi_dict:
        return True
    roi = roi_dict[site]

    x = float(marking["x"])
    y = float(marking["y"])


    X = []
    Y = []

    for segment_index in range(len(roi)-1):
        rX1,rY1 = roi[segment_index]
        X.append(rX1)
        Y.append(-rY1)

    # find the line segment that "surrounds" x and see if y is above that line segment (remember that
    # images are flipped)
    for segment_index in range(len(roi)-1):
        if (roi[segment_index][0] <= x) and (roi[segment_index+1][0] >= x):
            rX1,rY1 = roi[segment_index]
            rX2,rY2 = roi[segment_index+1]

            # todo - check why such cases are happening
            if rX1 == rX2:
                continue

            m = (rY2-rY1)/float(rX2-rX1)
            rY = m*(x-rX1)+rY1

            if y >= rY:
                # we have found a valid marking
                # create a special type of animal None that is used when the animal type is missing
                # thus, the marking will count towards not being noise but will not be used when determining the type

                return True
            else:
                return False

    # probably shouldn't happen too often but if it does, assume that we are outside of the ROI
    return False

# load the roi file - for checking if a given marking is within the ROI
with open("/home/ggdhines/github/Penguins/public/roi.tsv","rb") as roiFile:
    roiFile.readline()
    reader = csv.reader(roiFile,delimiter="\t")
    for l in reader:
        path = l[0]
        t = [r.split(",") for r in l[1:] if r != ""]
        roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]

client = pymongo.MongoClient()
db = client['penguin']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

# for c in classification_collection.find():
#     _id = c["_id"]
#     zooniverse_id = c["subjects"][0]["zooniverse_id"]
#
#     classification_collection.update_one({"_id":_id},{"$set":{"zooniverse_id":zooniverse_id}})

clustering_engine = Agglomerative(None,{})

# result = db.profiles.create_index([('zooniverse_id', pymongo.ASCENDING)],unique=False)
# print result

for c in classification_collection.find().limit(1000):
    _id = c["_id"]
    zooniverse_id = c["subjects"][0]["zooniverse_id"]
    # print(zooniverse_id)

    markings = []
    user_ids = []
    tools = []

    num_users = 0
    path = subject_collection.find_one({"zooniverse_id":zooniverse_id})["metadata"]["path"]
    _,image_id = path.split("/")

    site_id = image_id.split("_")[0]
    print(site_id)

    # print(path.split("/"))
    print(site_id in name_changes,site_id in roi_dict)
    print(name_changes)
    continue


    for c2 in classification_collection.find({"zooniverse_id":zooniverse_id}):
        num_users += 1
        if "finished_at" in c2["annotations"][1]:
            continue

        if "user_name" in c2:
            id_ = c2["user_name"]
        else:
            id_ = c2["user_ip"]

        try:
            for penguin in c2["annotations"][1]["value"].values():
                x = float(penguin["x"])
                y = float(penguin["y"])
                penguin_type = penguin["value"]

                markings.append((x,y))
                user_ids.append(id_)
                tools.append(penguin_type)
        except AttributeError:
            continue

    if markings != []:
        clustering_results = clustering_engine.__cluster__(markings,user_ids,tools,markings,None,None)

        for penguin_index,cluster in enumerate(clustering_results[0]):
            center = cluster["center"]
            tools = cluster["tools"]

            probability_adult = sum([1 for t in tools if t == "adult"])/float(len(tools))
            probability_true_positive = len(tools)/float(num_users)
            count_true_positive = len(tools)

            print(penguin_index,center[0],center[1],probability_adult,probability_true_positive,count_true_positive)
    else:
        print(-1,-1,-1,-1,-1,-1)
