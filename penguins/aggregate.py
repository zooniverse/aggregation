#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import pymongo
import sys
sys.path.append("/home/ggdhines/github/aggregation/engine")
from agglomerative import Agglomerative
import csv
import os
import psycopg2
from multiprocessing import Pool

name_changes = {}
with open("/home/ggdhines/Downloads/Nomenclature_changes.csv","rb") as f:
    f.readline()
    reader = csv.reader(f,delimiter=",")

    for zoo_id,pre_zoo_id in reader:
        # print(pre_zoo_id+"|")
        # print(pre_zoo_id == "")
        if pre_zoo_id != "":
            name_changes[zoo_id] = pre_zoo_id[:-1]

# assert False
roi_dict = {}


# method for checking if a given marking is within the ROI
def __in_roi__(site,marking):
    """
    does the actual checking
    :param object_id:
    :param marking:
    :return:
    """

    if site not in roi_dict:
        return True
    roi = roi_dict[site]

    x,y = marking

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

def dummy_func((zooniverse_id,markings,user_ids,tools)):
    if markings == []:
        return zooniverse_id,(-1,0)
    clustering_engine = Agglomerative(None,None,{})
    clustering_results = clustering_engine.__cluster__(markings,user_ids,tools,markings,None,None)
    return zooniverse_id,clustering_results

client = pymongo.MongoClient()
db = client['penguin']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

# for c in classification_collection.find():
#     _id = c["_id"]
#     zooniverse_id = c["subjects"][0]["zooniverse_id"]
#
#     classification_collection.update_one({"_id":_id},{"$set":{"zooniverse_id":zooniverse_id}})



# result = db.profiles.create_index([('zooniverse_id', pymongo.ASCENDING)],unique=False)
# print result

conn = psycopg2.connect("dbname='penguins' user='postgres' host='localhost' password='apassword'")
conn.autocommit = True
cur = conn.cursor()

pool = Pool(processes=4)

processes = []

for subject in subject_collection.find()[:10]:
    # _id = c["_id"]

    zooniverse_id = subject["zooniverse_id"]
    print(zooniverse_id)




    markings = []
    user_ids = []
    tools = []

    num_users = 0


    cur.execute("select user_id,annotations from classifications where zooniverse_id = '" + zooniverse_id + "'")
    for user_id,annotations in cur.fetchall():

        num_users += 1
        if "finished_at" in annotations:
            continue

        try:
            for penguin in annotations[1]["value"].values():
                if penguin["value"] == "other":
                    continue

                try:
                    x = float(penguin["x"])
                    y = float(penguin["y"])
                except TypeError:
                    print(penguin)
                    raise
                except ValueError:
                    print("skipping bad markings")
                    continue

                # if site_id in roi_dict:
                #     if not __in_roi__(site_id,(x,y)):
                #         print("skipping due to being outside roi")
                #         continue

                markings.append((x,y))
                user_ids.append(user_id)
                tools.append(penguin["value"])
        except AttributeError:
            continue
        except KeyError:
            continue


    processes.append(pool.apply_async(dummy_func,[(zooniverse_id,markings,user_ids,tools),]))


for p in processes:
    zooniverse_id,(clustering_results,_) = p.get()

    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})

    path = subject["metadata"]["path"]
    _,image_id = path.split("/")

    site_id = image_id.split("_")[0]
    # print(site_id)

    big_path,little_path = path.split("/")
    little_path = little_path[:-4]

    d = "/tmp/penguin/"+big_path
    if not os.path.exists(d):
        os.makedirs(d)

    with open(d+"/"+little_path+".csv","w") as f:
        if clustering_results == -1:
            f.write("-1\n")
        else:
            f.write("penguin_index,x_center,y_center,probability_of_adult,probability_of_chick,probability_of_egg,probability_of_true_positive,num_markings\n")

            for penguin_index,cluster in enumerate(clustering_results):
                center = cluster["center"]
                tools = cluster["tools"]

                probability_adult = sum([1 for t in tools if t == "adult"])/float(len(tools))
                probability_chick = sum([1 for t in tools if t == "chick"])/float(len(tools))
                probability_egg = sum([1 for t in tools if t == "egg"])/float(len(tools))
                probability_true_positive = len(tools)/float(num_users)
                count_true_positive = len(tools)

                f.write(str(penguin_index)+","+str(center[0])+","+str(center[1])+","+str(probability_adult)+","+str(probability_chick)+"," + str(probability_egg)+ ","+str(probability_true_positive)+","+str(count_true_positive)+"\n")
