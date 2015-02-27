#!/usr/bin/env python
import pymongo
from shapely.geometry import Polygon
from shapely.geos import TopologicalError
import matplotlib.pyplot as plt
from shapely.validation import explain_validity
import math

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
    correct = []
    to_process = plist

    while to_process != []:
        for j in range(len(to_process),3,-1):
            shape1 = Polygon(to_process[:j])
            validity = explain_validity(shape1)
            if validity == "Valid Geometry":
                correct.append(to_process[:j])
                to_process = to_process[j:]
                break


            # if validity == "Valid Geometry":
            #     valid = True
            # else:
            #     if valid:
            #         valid = False
            #         breaks.append(j)

        print breaks

        print

        return []



        if "[" in validity:
            print validity
            print p
            # there is a self loop - so break the list up
            s,t = validity.split(" ")
            q,r = s.split("[")
            r,t =  float(r), float(t[:-1])

            splits = [i for i in range(len(p)) if math.sqrt((p[i][0]-r)**2 + (p[i][1]-t)**2) < 0.01]
            print splits
            X,Y = zip(*p)
            plt.plot(X,Y)
            plt.show()
            assert False
            for jj in range(len(splits)-1):
                temp = p[splits[jj]:splits[jj+1]]
                if len(temp) > 1:
                    plist.append(temp[:])

            temp = p[splits[-1]:]
            temp.extend(p[:splits[0]])
            if len(temp) > 1:
                plist.append(temp[:])
        else:
            correct.append(p[:])

    return correct

for counter,classification in enumerate(classification_collection.find().limit(10000)):
    #print counter
    annotations = classification["annotations"]
    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    if "user_name" in classification:
        user = classification["user_name"]
    else:
        user = classification["user_ip"]

    index = [("value" in d) for d in annotations].index(True)

    polygons = annotations[index]["value"]
    if polygons == "":
        continue

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

            x += dX
            y += dY

            plines.append((x,y))

        if not(zooniverse_id in polydict):
            polydict[zooniverse_id]= {}
            polydict[zooniverse_id][user] = fix_poly(plines)
        else:
            if not(user in polydict[zooniverse_id]):
                polydict[zooniverse_id][user] = fix_poly(plines)
            else:
                polydict[zooniverse_id][user].extend(fix_poly(plines))



def merge(plist1,plist2):
    for p1 in plist1:
        # while p1[-1] in p1[:-1]:
        #     p1.pop(-1)
        #p1 = [p1[j] for j in range(len(p1)) if not(p1[j] in p1[:j])]
        # print len(list(set(p1))), len(p1)

        shape1 = Polygon(p1)
        validity = explain_validity(shape1)
        if "[" in validity:
            s,t = validity.split(" ")
            q,r = s.split("[")
            r,t =  float(r), float(t[:-1])

            print len(p1)
            print p1
            print r,t
            print
            print
            splits = [i for i in range(len(p1)) if math.sqrt((p1[i][0]-r)**2 + (p1[i][1]-t)**2) < 0.01]

            for jj in range(len(splits)-1):

                print p1[splits[jj]:splits[jj+1]]

            print p1[:splits[0]]
            print p1[splits[-1]:]
            print
            assert False

        for p2 in plist2:
            #print len(list(set(p2))), len(p2)
            #p2 = [p2[j] for j in range(len(p2)) if not(p2[j] in p2[:j])]
            # while p2[-1] in p2[:-1]:
            #     p2.pop(-1)

            shape2 = Polygon(p2)
            try:
                if shape1.intersects(shape2):
                    shape3 = shape1.intersection(shape2)
                    print shape3.area

                    X,Y = zip(*p2)
                    plt.plot(X,Y,color="black")

                    X,Y = zip(*p1)
                    plt.plot(X,Y,color="black")

                    plt.show()
            except TopologicalError:
                X,Y = zip(*p2)
                plt.plot(X,Y,color="green")

                X,Y = zip(*p1)
                plt.plot(X,Y,color="red")
                plt.show()

for zooniverse_id in count:
    if count[zooniverse_id] == 1:
        continue

    #shapes = {}
    #for users in polydict[zooniverse_id]:
    #    shapes[users] = [Polygon(p) for p in polydict[zooniverse_id][users]]

    u = polydict[zooniverse_id].keys()
    u0 = u[0]
    u1 = u[1]

    merge(polydict[zooniverse_id][u0],polydict[zooniverse_id][u1])