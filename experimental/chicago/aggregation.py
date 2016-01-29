#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import matplotlib.cbook as cbook
import random
import bisect

animals = [u'bike', u'grayFox', u'livestock', u'foxSquirrel', u'deer', u'rat', u'mink', u'human', u'beaver', u'mouse', u'muskrat', u'domDog', u'mower', u'graySquirrel', u'opossum', u'domCat', u'chipmunk', u'bird', u'otherVehicle', u'redFox', u'horse', u'woodChuck', u'rabbit', u'coyote', u'car', u'flyingSquirrel', u'melanisticGraySquirrel', u'raccoon', u'skunk']
print len(animals)


results = {}

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc



client = pymongo.MongoClient()
db = client['chicago_2015-01-04']
classification_collection = db["chicago_classifications"]

ip_listing = []

#the header for the csv input file
f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")

subject_list = []
user_list = []

for classification in classification_collection.find():
    if classification["subjects"] == []:
        continue

    zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    if "user_name" in classification:
        user = classification["user_name"]
    else:
        user = classification["user_ip"]

    try:
        user_index = index(user_list,user)
    except ValueError:
        bisect.insort(user_list,user)

    try:
        subject_index = index(subject_list,zooniverse_id)
    except ValueError:
        bisect.insort(subject_list,zooniverse_id)

print "****"

f_ibcc = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f_ibcc.write("a,b,c\n")

for classification in classification_collection.find():
    if classification["subjects"] == []:
        continue

    if classification["tutorial"] is True:
        continue

    zooniverse_id = str(classification["subjects"][0]["zooniverse_id"])

    if not(zooniverse_id in results):
        results[zooniverse_id] = dict.fromkeys(animals,0)

    if "user_name" in classification:
        user = classification["user_name"]
    else:
        user = classification["user_ip"]

    try:
        user_index = user_list.index(user)
    except ValueError:
        user_list.append(user)
        user_index = len(user_list) - 1

    subject_index = index(subject_list,zooniverse_id)
    user_index = index(user_list,user)

    if "finished_at" in classification["annotations"][0]:
        continue
    else:
        species = classification["annotations"][0]["species"]
        if len(classification["annotations"]) != 4:
            print classification["annotations"]
        results[zooniverse_id][species] += 1
        #print results[zooniverse_id]


    continue
    #print classification["annotations"][0]

    mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
    markings = classification["annotations"][mark_index].values()[0]

    found_condor = "0"
    for animal in markings.values():
        try:
            animal_type = animal["species"]
        except KeyError:
            continue
        print animal_type
        break

    f_ibcc.write(str(user_index)+","+str(subject_index)+","+found_condor+"\n")

print results
assert(False)

with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array([30,70])\n")
    f.write("alpha0 = np.array([[3, 1], [1,3]])\n")



#start by removing all temp files
try:
    os.remove(base_directory+"/Databases/condor_ibcc.out")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.csv.dat")
except OSError:
    pass

ibcc.runIbcc(base_directory+"/Databases/condor_ibcc.py")

#read in the results
# with open(base_directory+"/Databases/condor_ibcc.out","rb") as f:
#     reader = csv.reader(f,delimiter=" ")
#
#     for subject_index,p0,p1 in reader:
#         subject_index = int(float(subject_index))
#         print p1

#now repeat - but with fewer users per image
f.close()
f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")
for subject_count,zooniverse_id in enumerate(sampled_ids):
    user_ips_to_sample = []
    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        user_ips_to_sample.append(classification["user_ip"])


    sample = random.sample(user_ips_to_sample,2)
    for user_ip in sample:
        classification = classification_collection.find_one({"subjects.zooniverse_id":zooniverse_id,"user_ip":user_ip})

        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            found_condor = "0"
            for animal in markings.values():
                try:
                    animal_type = animal["animal"]
                except KeyError:
                    continue
                if animal_type == "condor":
                    found_condor = "1"
                    break


            f.write(str(str(ip_listing.index(user_ip)))+","+str(subject_count)+","+found_condor+"\n")
        except ValueError:
            pass

with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out2\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array([30,70])\n")
    f.write("alpha0 = np.array([[3, 1], [1,3]])\n")



#start by removing all temp files
try:
    os.remove(base_directory+"/Databases/condor_ibcc.out2")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.csv.dat")
except OSError:
    pass

ibcc.runIbcc(base_directory+"/Databases/condor_ibcc.py")


