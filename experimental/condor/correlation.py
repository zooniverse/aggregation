#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import matplotlib.pyplot as plt

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"


sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-11']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]


to_sample_from = [u'ACW0000alg', u'ACW0005bd8', u'ACW0000fcd', u'ACW0004ab2', u'ACW0000k1n', u'ACW00012zi', u'ACW00001v6', u'ACW0000brn', u'ACW00015ip', u'ACW0004mjp', u'ACW0004h4h', u'ACW000422b', u'ACW0000c8c', u'ACW0003v8i', u'ACW00012v0', u'ACW000313b', u'ACW0000pks', u'ACW0000dyy', u'ACW0004yfc', u'ACW0000azk', u'ACW00013un', u'ACW0004wam', u'ACW000163t', u'ACW00014qq', u'ACW0004n7t', u'ACW0000lk6', u'ACW0005kfq', u'ACW0000o33', u'ACW0000zkx', u'ACW00034vs', u'ACW00006qd', u'ACW0000bnm', u'ACW00012mu', u'ACW00012ee', u'ACW0004fc2', u'ACW0000t98', u'ACW0004xen', u'ACW00011jd', u'ACW00043a0', u'ACW0004vxm', u'ACW0000pqj', u'ACW0003z3f', u'ACW000113n', u'ACW0000c5g', u'ACW00003hg', u'ACW00006st', u'ACW000052d', u'ACW00008rk', u'ACW00030pt', u'ACW0005fg8', u'ACW0004bej', u'ACW00033em', u'ACW00040nd', u'ACW0001ork', u'ACW00041ln', u'ACW0000lgo', u'ACW0000520', u'ACW000095e', u'ACW00012wa', u'ACW0003v6k', u'ACW0000pyr', u'ACW000039w', u'ACW0005abs', u'ACW0000e2t', u'ACW00015qr', u'ACW000094k', u'ACW0004vpd', u'ACW0003f2p', u'ACW0003onz', u'ACW0000az7', u'ACW000468z', u'ACW00033mp', u'ACW0004uuu', u'ACW0003pco', u'ACW0003kb2', u'ACW00004uc', u'ACW0004cfd', u'ACW0005kg1', u'ACW0004jhf', u'ACW00053os', u'ACW00058gi', u'ACW0000de8', u'ACW00006w3', u'ACW00013o0', u'ACW00006oy', u'ACW0001x39', u'ACW00017qc', u'ACW00059rz', u'ACW00043gl', u'ACW00011n2', u'ACW0000o2m', u'ACW0000vy6', u'ACW00007cx', u'ACW0000cqs', u'ACW0002zz9', u'ACW00012br', u'ACW0000l61', u'ACW0000kzz', u'ACW0004ecw', u'ACW0001x6y', u'ACW0003w2u', u'ACW00055xe', u'ACW0003w5m', u'ACW0000lrf', u'ACW00013la', u'ACW0004r0g', u'ACW0004rpf', u'ACW0000psb', u'ACW0004j9c', u'ACW00003c6', u'ACW0000p8k', u'ACW0002hkm', u'ACW00006jm', u'ACW00006sc', u'ACW00012pg', u'ACW000113q', u'ACW0003yhf', u'ACW000471i', u'ACW0004tgw', u'ACW00011e7', u'ACW0000y4f', u'ACW00011r3', u'ACW0000ld8', u'ACW0005eqd', u'ACW0004jaz', u'ACW00041sq', u'ACW00008t4', u'ACW0003ex5', u'ACW0000s9l', u'ACW00012ga', u'ACW0003rgk', u'ACW000111p', u'ACW0000pjp', u'ACW0005dv0', u'ACW0000mhg', u'ACW0004hsy', u'ACW0002qq5', u'ACW0004dnv', u'ACW00002zj', u'ACW0000biu', u'ACW0000bsc', u'ACW0000dzs', u'ACW0000lbe', u'ACW00041nj', u'ACW0002g6e', u'ACW0000p8b', u'ACW0000s5q', u'ACW0005etp', u'ACW00012yz', u'ACW0003kyk', u'ACW00005ci', u'ACW0000tsa', u'ACW0003s2z', u'ACW0000b4p', u'ACW00013f2', u'ACW00017a5', u'ACW0004dyp', u'ACW000031s', u'ACW00003m1', u'ACW0000prs', u'ACW00045ie', u'ACW0004vkd', u'ACW0002l3e', u'ACW00011a4', u'ACW00011kg', u'ACW00016km', u'ACW0004ph0', u'ACW0005ghu', u'ACW00006pw', u'ACW0000e56', u'ACW0003vcq', u'ACW00052cu', u'ACW00040f6', u'ACW0000mdg', u'ACW00053of', u'ACW0000oub', u'ACW000095h', u'ACW0000m2v', u'ACW00003ii', u'ACW0004hse', u'ACW00055tl', u'ACW00057jb', u'ACW0005kf4', u'ACW0004c1v', u'ACW0001281', u'ACW0004ioq', u'ACW000095i']
to_ignore = ["ACW0000t98","ACW0005abs","ACW0000e2t","ACW0000az7","ACW00007cx","ACW0000p8k","ACW0000tsa","ACW0000oub","ACW0000m2v","ACW0004ecw","ACW0005ghu","ACW00052cu"]
steps = [2,5,20]
condor_count_2 =  {k:[] for k in steps}
condor_count_3 =  {k:[] for k in steps}

big_userList = {}
animal_count = 0
names = [u'itsmestephanie', u'stonepenny',  u'miltonbosch', u'wreness']
Y  = []
X =[]
for subject_count,zooniverse_id in enumerate(to_sample_from):
    if zooniverse_id in to_ignore:
        continue
    print zooniverse_id
    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})
    url = subject["location"]["standard"]

    slash_index = url.rfind("/")
    object_id = url[slash_index+1:]


    annotation_list = []

    user_markings = []
    user_list = []
    type_list = []

    numAnimals = []

    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if not(user in big_userList):
            big_userList[user] = 1
        else:
            big_userList[user] += 1

        if user in user_list:
            continue

        numAnimals = 0

        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            for animal in markings.values():
                scale = 1.875
                x = scale*float(animal["x"])
                y = scale*float(animal["y"])

                try:
                    animal_type = animal["animal"]
                    if animal_type in ["condor","turkeyVulture","goldenEagle"]:
                        user_markings.append((x,y))
                        user_list.append(user)
                        type_list.append(animal_type)

                        numAnimals += 1

                except KeyError:
                    pass

        except ValueError:
            pass

        if user in names:
            print user,numAnimals

    Y.append(np.median(numAnimals))

    identified_animals,clusters = DivisiveKmeans(3).fit2(user_markings,user_list,debug=True)
    identified_animals,clusters = DivisiveKmeans(3).__fix__(identified_animals,clusters,annotation_list,user_list,200)
    print len(identified_animals)
    X.append(len(identified_animals))

#print sorted(big_userList.items(),key = lambda x:x[1])
#plt.plot(X,Y,'.')
#plt.show()


