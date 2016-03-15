__author__ = 'ggdhines'
import pymongo
import csv
import cPickle as pickle
import os

client = pymongo.MongoClient()
db = client['penguin_2015-05-08']
classification_collection = db["penguin_classifications"]
subject_collection = db["penguin_subjects"]

# load rois
roi_dict = {}

with open("/home/ggdhines/github/Penguins/public/roi.tsv","rb") as roi_file:
    roi_file.readline()
    reader = csv.reader(roi_file,delimiter="\t")
    for l in reader:
        path = l[0]
        t = [r.split(",") for r in l[1:] if r != ""]
        roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]


markings_dict = dict()
gold_dict = dict()

for c in classification_collection.find({"user_name":"caitlin.black"})[:25]:
    zooniverse_id = c["subjects"][0]["zooniverse_id"]
    print zooniverse_id

    subject =  subject_collection.find_one({"zooniverse_id":zooniverse_id})
    try:
        animals_present = subject["metadata"]["counters"]["animals_present"]
    except KeyError:
        continue

    if animals_present < 5 :
        continue

    file_base = "/home/ggdhines/Databases/penguin/"+zooniverse_id
    if os.path.isfile(file_base+".pickle"):
        markings_dict[zooniverse_id] = pickle.load(open(file_base+".pickle","rb"))
        gold_dict[zooniverse_id] = pickle.load(open(file_base+"_gold.pickle","rb"))
    else:
        markings_dict[zooniverse_id] = dict()


        for c2 in classification_collection.find({"subjects" : {"$elemMatch": {"zooniverse_id":zooniverse_id}}}):
            user_ip = c2["user_ip"]
            m = []
            # markings_dict[zooniverse_id][user_ip] = []

            try:
                markings_list = c2["annotations"][1]["value"]
                if isinstance(markings_list,dict):
                    for marking in markings_list.values():
                        if marking["value"] in ["adult","chick"]:
                            x,y = (float(marking["x"]),float(marking["y"]))
                            # markings_dict[zooniverse_id][user_ip].append((x,y))
                            m.append((x,y))

                if ("user_name" in c2) and (c2["user_name"] == "caitlin.black"):
                    gold_dict[zooniverse_id] = m
                    pickle.dump(m,open(file_base+"_gold.pickle","wb"))
                else:
                    markings_dict[zooniverse_id][user_ip] = m
            except KeyError:
                continue

        pickle.dump(markings_dict[zooniverse_id],open(file_base+".pickle","wb"))



