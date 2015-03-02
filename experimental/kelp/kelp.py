#!/usr/bin/env python
import pymongo
from shapely.geometry import Polygon
from shapely.geos import TopologicalError
import matplotlib.pyplot as plt
from shapely.validation import explain_validity
import math
import os
import cPickle as pickle

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
    github_directory = base_directory + "/github"
    code_directory = base_directory + "/PycharmProjects"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"
    github_directory = base_directory +"/github"
    print github_directory
else:
    base_directory = "/home/greg"
    code_directory = base_directory + "/github"
    github_directory = base_directory + "/github"


project = "kelp"
date = "2015-02-22"

client = pymongo.MongoClient()
db = client[project+"_"+date]
classification_collection = db[project+"_classifications"]
subject_collection = db[project+"_subjects"]
user_collection = db[project+"_users"]

polydict = {}
count = {}

def fix_poly(plist):
    print "  " + str(len(plist))
    try:
        shape = Polygon(plist)
    except ValueError:
        return []

    validity = explain_validity(shape)
    if validity == "Valid Geometry":
        return [shape]

    shape_list = []
    while plist != []:
        #print "  " + str(len(plist))
        longest_valid = 0
        best_choice = None
        for i in range(len(plist)):
            # it might be that no matter how long the longest valid polygon is from this point, it can't
            # beat the current max, if so, just break
            if (len(plist)-i) < longest_valid:
                break

            longest_valid_i = 0
            best_choice_i = None
            for j in range(i+1,len(plist)):
                try:
                    temp_shape = Polygon(plist[i:j])
                    validity = explain_validity(temp_shape)

                    if (validity == "Valid Geometry") and ((j-i) > longest_valid_i):
                        longest_valid_i = j-i
                        best_choice_i = i,j
                except ValueError:
                    continue
            if longest_valid_i > longest_valid:
                longest_valid = longest_valid_i
                best_choice = best_choice_i
        if best_choice is None:
            break

        i,j = best_choice
        shape = Polygon(plist[i:j])
        shape_list.append(shape)
        test_validity = explain_validity(shape)
        assert test_validity == "Valid Geometry"

        plist_temp = plist[:i]
        plist_temp.extend(plist[j:])
        plist = plist_temp
    return shape_list
counter = -1
for classification in classification_collection.find():
    counter += 1
    if counter == 10000:
        break

    annotations = classification["annotations"]
    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    if "user_name" in classification:
        user_id = classification["user_name"]
        user = user_collection.find_one({"name":user_id})
    else:
        user_id = classification["user_ip"]
        user = user_collection.find_one({"ip":user_id})

    if user is None:
        continue
    zooniverse_user_id = user["zooniverse_id"]
    index = [("value" in d) for d in annotations].index(True)

    polygons = annotations[index]["value"]
    if polygons == "":
        continue

    print counter

    # have we read in this user and subject before? If so, read in the pickle file
    fname = base_directory+"/Databases/kelp/"+str(zooniverse_id)+"_"+str(zooniverse_user_id)+".pickle"
    plines = []
    if os.path.isfile(fname):
        plines = pickle.load(open(fname,"rb"))
    else:

        if not(zooniverse_id in count):
            count[zooniverse_id] = 1
        else:
            count[zooniverse_id] += 1

        for poly_key in polygons:
            # convert from relative coordinates to absolute
            poly = polygons[poly_key]
            x,y = poly["startingPoint"]
            x = float(x)
            y = float(y)

            segmentIndices = sorted([int(i) for i in poly["relPath"]])
            plines = [(x,y)]
            for i in segmentIndices:
                delta = poly["relPath"][str(i)]
                dX,dY = delta
                dX = float(dX)
                dY = float(dY)

                if (dX == 0) and (dY == 0):
                    continue

                x += dX
                y += dY

                plines.append((x,y))

        plines = fix_poly(plines)
        pickle.dump(plines,open(fname,"wb"))

    if not(zooniverse_id in polydict):
        polydict[zooniverse_id]= {}
        polydict[zooniverse_id][user_id] = plines
    else:
        if not(user_id in polydict[zooniverse_id]):
            polydict[zooniverse_id][user_id] = plines
        else:
            print "weird"
            #polydict[zooniverse_id][user_id].extend(plines)



def intersect(plist1,plist2):
    for p1 in plist1:
        for p2 in plist2:
            shape = p1.intersection(p2)


print "here"

for zooniverse_id in count:
    if count[zooniverse_id] == 1:
        continue

    print zooniverse_id
    continue

    #shapes = {}
    #for users in polydict[zooniverse_id]:
    #    shapes[users] = [Polygon(p) for p in polydict[zooniverse_id][users]]

    u = polydict[zooniverse_id].keys()
    u0 = u[0]
    u1 = u[1]

    intersect(polydict[zooniverse_id][u0],polydict[zooniverse_id][u1])