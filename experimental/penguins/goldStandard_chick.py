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
    try:
        with open(base_directory + "/Databases/CEBannotations/"+site_name+"_chick_RAW.csv","rb") as f:
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





                    i0 = i1
    except IOError:
        sys.stderr.write("no file: " + base_directory + "/Databases/CEBannotations/"+site_name+"_chick_RAW.csv" +"\n")



