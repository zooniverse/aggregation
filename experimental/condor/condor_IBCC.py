#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys

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


to_sample_from = [u'ACW0000kxt', u'ACW00006p8', u'ACW0000bxt', u'ACW0002005', u'ACW000120u', u'ACW00006rc', u'ACW00040az', u'ACW0000m08', u'ACW0000az7', u'ACW000055u', u'ACW0000df0', u'ACW00006ld', u'ACW00011nb', u'ACW000180h', u'ACW0000k15', u'ACW0005ghc', u'ACW0000bl4', u'ACW00013hc', u'ACW0002t1k', u'ACW0000cu2', u'ACW00014ia', u'ACW00003ac', u'ACW00014vp', u'ACW0000nkd', u'ACW0003nyl', u'ACW0004k9y', u'ACW00012q9', u'ACW00011yg', u'ACW0000ozm', u'ACW00011hz', u'ACW000128j', u'ACW00006k6', u'ACW00012ha', u'ACW00007dn', u'ACW0004bp1', u'ACW00044cs', u'ACW0000lrr', u'ACW00015xo', u'ACW0000ddn', u'ACW0002g7h', u'ACW00053o5', u'ACW000127z', u'ACW0003zyk', u'ACW0001826', u'ACW0001evk', u'ACW0004feb', u'ACW0000jql', u'ACW0001hpb', u'ACW0000kw0', u'ACW00011gq', u'ACW00004vc', u'ACW00047sq', u'ACW000554b', u'ACW000181m', u'ACW0000k7q', u'ACW0000e6i', u'ACW0004jxu', u'ACW00011is', u'ACW00027lo', u'ACW0000lu1', u'ACW000130c', u'ACW0000le4', u'ACW000160y', u'ACW00051os', u'ACW0003y9q', u'ACW0004nra', u'ACW0002vj8', u'ACW00041en', u'ACW00057p7', u'ACW0002qps', u'ACW0000apl', u'ACW00007cw', u'ACW00018m9', u'ACW0005m6l', u'ACW00055cy', u'ACW00012xz', u'ACW0003yd6', u'ACW0000xdt', u'ACW0000pd9', u'ACW00003tq', u'ACW00011g4', u'ACW0000bv7', u'ACW00010ol', u'ACW000491z', u'ACW0000xf4', u'ACW000116t', u'ACW00002r7', u'ACW0000jw1', u'ACW00009lo', u'ACW000410t', u'ACW00003l5', u'ACW0002izy', u'ACW0000jt4', u'ACW00043gl', u'ACW00011wh', u'ACW0000ao8', u'ACW00048dl', u'ACW000036e', u'ACW0000m4n', u'ACW0003skl', u'ACW0000ijv', u'ACW0004s2k', u'ACW00011hn', u'ACW0000a2d', u'ACW0005ds7', u'ACW000138e', u'ACW0002sgv', u'ACW00006mc', u'ACW0003tvy', u'ACW000191i', u'ACW000037x', u'ACW0001sz7', u'ACW0004p03', u'ACW00003th', u'ACW00011ey', u'ACW0005e1z', u'ACW00008ax', u'ACW0003k73', u'ACW0000o4m', u'ACW00012gy', u'ACW00012j5', u'ACW0004iml', u'ACW0005anw', u'ACW0000jkb', u'ACW0000b4c', u'ACW0004tvd', u'ACW0000569', u'ACW00016p6', u'ACW0005f1n', u'ACW0005f5w', u'ACW0000lsm', u'ACW00003km', u'ACW0004e2v', u'ACW0004dt0', u'ACW00041nj', u'ACW0000396', u'ACW00013ni', u'ACW0003uar', u'ACW0005ck9', u'ACW0000dd6', u'ACW0004mno', u'ACW00007b9', u'ACW0005n2h', u'ACW00011di', u'ACW00033m4', u'ACW00006jl', u'ACW0000at6', u'ACW0000e13', u'ACW0001612', u'ACW0004e6m', u'ACW000030f', u'ACW0000xfq', u'ACW00012ag', u'ACW00033em', u'ACW0000aw8', u'ACW00011js', u'ACW0000auq', u'ACW0001235', u'ACW0004qkt', u'ACW0000s1g', u'ACW0000mac', u'ACW00011zg', u'ACW00013mn', u'ACW0000ms9', u'ACW0004ijh', u'ACW0005ff4', u'ACW00011na', u'ACW0000pd3', u'ACW0001234', u'ACW00057hs', u'ACW0000lr6', u'ACW0000kko', u'ACW0004s6n', u'ACW0001b1c', u'ACW0003v83', u'ACW000138l', u'ACW000030u', u'ACW0000boq', u'ACW00047pv', u'ACW00054bm', u'ACW0004ehj', u'ACW0000b8l', u'ACW0003s9d', u'ACW00003b2', u'ACW00041cn', u'ACW0000dxs', u'ACW00011qs', u'ACW0004leg', u'ACW00012t3', u'ACW0000arl', u'ACW0005ev1', u'ACW00039vc', u'ACW0001t23', u'ACW0000jxm', u'ACW0003c0h', u'ACW00041ba', u'ACW0003v1j', u'ACW00011j7', u'ACW0000nyy', u'ACW0000br8', u'ACW0000xe4', u'ACW000460a', u'ACW0004ezy', u'ACW00003jx']
to_ignore_1 = ["ACW0002005","ACW0000m08","ACW0000az7","ACW000055u","ACW00014vp","ACW0000nkd","ACW0003nyl","ACW0000jql","ACW0000k7q","ACW0000e6i","ACW0000lu1","ACW0002qps","ACW00003tq","ACW00009lo","ACW0000jt4","ACW0000m4n","ACW00003th","ACW0000o4m","ACW00033m4","ACW0000s1g","ACW0000pd3","ACW0000kko","ACW00039vc","ACW0003c0h"]
to_ignore_2 = ["ACW0004feb","ACW0002vj8","ACW00012xz","ACW0000pd9","ACW0000xf4","ACW0002izy","ACW0000569","ACW0000dd6","ACW0000at6","ACW0001b1c","ACW0001t23","ACW00003jx"]

steps = [2,5,20]
condor_count_2 =  {k:[] for k in steps}
condor_count_3 =  {k:[] for k in steps}

big_userList = []
animal_count = 0

f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")

for subject_count,zooniverse_id in enumerate(to_sample_from):
    if zooniverse_id in to_ignore_2:
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

    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if not(user in big_userList):
            big_userList.append(user)

        if user in user_list:
            continue

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

                except KeyError:
                    pass

        except ValueError:
            pass


    identified_animals,clusters = DivisiveKmeans(1).fit2(user_markings,user_list,debug=True)

    for animal_cluster in clusters:
        #print "===="
        animal_count += 1
        for pt in animal_cluster:
            #convert user id into global index
            user_index = big_userList.index(user_list[user_markings.index(pt)])
            user_identified = type_list[user_markings.index(pt)]
            if user_identified == "condor":
                f.write(str(user_index) + ","+str(animal_count) + ",1\n")
                #print str(user_index) + ","+str(animal_count) + ",1"
            else:
                f.write(str(user_index) + ","+str(animal_count) + ",0\n")
                #print str(user_index) + ","+str(animal_count) + ",0"


f.close()
with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    #f.write("nu0 = np.array([30,70])\n")
    #f.write("alpha0 = np.array([[3, 1], [1,3]])\n")



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


