#!/usr/bin/env python
__author__ = 'greghines'
import pymongo
import re
import os
import sys

client = pymongo.MongoClient()
db = client['penguin_2014-09-27']
collection = db["penguin_subjects"]

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

#has changed - MAIVb2013,
#issues with -
for site_name in ["GEORa2013b","LOCKb2013b","MAIVc2013","PETEa2012a","PETEd2013a","PETEd2013b","PETEe2013b","SALIa2012a","YALOa2013a","YALOa2014a"]:
    #print site_name
    sys.stderr.write(str(site_name)+"\n")
    with open(base_directory + "/Databases/CEBannotations/"+site_name+"_adult_RAW.csv","rb") as f:
    #with open("/home/greg/Downloads/MAIVb2013_adult_RAW.csv","rb") as f:
            l = f.readline()
            i0 = 0
            counter = 0
            while True:
                if i0 == -1:
                    break



                http = l.find("http",i0)
                JPG = l.find("JPG",i0)
                url = l[http:JPG+3]
                #print url
                #break

                slashIndex = url.rfind("/")
                id_ = url[slashIndex+1:-4]



                r = collection.find_one({"metadata.path": {"$regex": id_}})

                i1 = l.find("http",JPG)
                if r is not None:

                    classification_count = r["classification_count"]
                    assert(isinstance(classification_count,int))
                    if classification_count >= 10:
                        line_out = r["zooniverse_id"] + "\t"
                        markings = l[JPG+5:i1-3].split(";")
                        #if at least one marking was made
                        if markings != [""]:
                            for m in markings:
                                x,y,temp = m.split(",")
                                line_out += x+","+y+";"

                            print line_out
                            counter += 1
                            sys.stderr.write(str(counter)+"\n")


                #continue

                #if counter == 2:
                #    break

                # print url
                # #for r in collection.find({"metadata.path": {"$regex": number}}):
                # r = collection.find_one({"$and": [{"metadata.path": {"$regex": number}},{"metadata.path": {"$regex": "MAIV"}}]})
                # print r["metadata"]["path"]
                # print r["location"]["standard"]
                # found = True
                #
                # if found:
                #     break



                i0 = i1


                # i2 = l.find("\"",i1+1)
                #
                #
                # i0 = i2 + 1
                # annotations = l[i1+1:i2]
                #
                # if r is not None:
                #     pts = r["zooniverse_id"] + "\t"
                #     for a in  annotations.split(";")[:-1]:
                #         pts += a.split(',')[0]
                #         pts += ","
                #         pts += a.split(',')[1]
                #         pts += ";"
                #
                #     print pts


