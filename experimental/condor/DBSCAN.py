#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import csv
import sys
import os
import pymongo
import urllib
import matplotlib.cbook as cbook

sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
import adaptiveDBSCAN


client = pymongo.MongoClient()
db = client['condor_2014-09-19']
collection = db["condor_classifications"]

toSkip = ["ACW00007l2","ACW00008sh","ACW0001828","ACW00013hm","ACW0000m44","ACW000184q"]
toProcess = [u'ACW0000gvo', u'ACW0000xby', u'ACW0000x0h', u'ACW0000w7s', u'ACW000169y', u'ACW0000y5t', u'ACW000052d', u'ACW0000tgh', u'ACW0001828', u'ACW0000lyk', u'ACW0000wm3', u'ACW0000xqt', u'ACW00012o9', u'ACW0000y58', u'ACW0000lmz', u'ACW0000al9', u'ACW0000bzq', u'ACW0000mow', u'ACW00011uw', u'ACW00014fq', u'ACW0000s5k', u'ACW0000r7z', u'ACW00010p0', u'ACW00017ns', u'ACW0000jbl', u'ACW0000lkv', u'ACW00015w1', u'ACW0000kdn', u'ACW00007pw', u'ACW0000ztc', u'ACW000029k', u'ACW00010wr', u'ACW0000e4v', u'ACW0000w4h', u'ACW00017au', u'ACW0000xan', u'ACW0000we9', u'ACW0000fjs', u'ACW00006ld', u'ACW00012h1', u'ACW00014i1', u'ACW0000owj', u'ACW00001qp', u'ACW00001uo', u'ACW00017gb', u'ACW00015si', u'ACW0000jzl', u'ACW00003ku', u'ACW00017l4', u'ACW00014r5']
toProcess = [u'ACW000181g', u'ACW0000tle', u'ACW00008pl', u'ACW0000fgt', u'ACW0000gvo', u'ACW00014fq', u'ACW00003b2', u'ACW00008y9', u'ACW0000w7s', u'ACW00013s1', u'ACW000169y', u'ACW0000whb', u'ACW000185d', u'ACW0000leq', u'ACW0000c0o', u'ACW00001y5', u'ACW00015qn', u'ACW00008pw', u'ACW00008ok', u'ACW000151g', u'ACW0000lmz', u'ACW0000al9', u'ACW00005j2', u'ACW0000jx4', u'ACW0000mow', u'ACW0004dnv', u'ACW00003hv', u'ACW0000qcz', u'ACW0000zgo', u'ACW00008i3', u'ACW00015rk', u'ACW0000r7z', u'ACW0001420', u'ACW00010p0', u'ACW000026z', u'ACW000128x', u'ACW0000uc8', u'ACW0000wiw', u'ACW0000ss4', u'ACW00003e1', u'ACW0000amf', u'ACW0000ztc', u'ACW00008o6', u'ACW0000tjg', u'ACW0000z1w', u'ACW00014zh', u'ACW00013sv', u'ACW0000e2b', u'ACW0000mg6', u'ACW00012n2', u'ACW00013sn', u'ACW0000ga8', u'ACW0000gqf', u'ACW0000y22', u'ACW00002ya', u'ACW0000xoo', u'ACW0000rai', u'ACW0000alz', u'ACW00012ed', u'ACW0000ujo', u'ACW000184q', u'ACW00002rw', u'ACW00014i1', u'ACW00001qp', u'ACW00001qs', u'ACW0000sk1', u'ACW00015si', u'ACW0001280', u'ACW0000qg8', u'ACW0000s70', u'ACW00017tb', u'ACW0000dxa', u'ACW00012ez', u'ACW0000sa3', u'ACW0000q9t', u'ACW0000k37', u'ACW0000r0n', u'ACW0000w49', u'ACW0000ff0', u'ACW0000axn', u'ACW0000tkj', u'ACW0000ouv', u'ACW0000tri', u'ACW000042a', u'ACW00012ph', u'ACW0000dmc', u'ACW00001uo', u'ACW0001828', u'ACW000052d', u'ACW00008f3', u'ACW00012o9', u'ACW000029k', u'ACW00007g9', u'ACW00016he', u'ACW0000hrt', u'ACW00011uw', u'ACW0000pt0', u'ACW00014oq', u'ACW00007pe', u'ACW0003wmn', u'ACW0000pjt', u'ACW0000fih', u'ACW0000p5w', u'ACW0000s5k', u'ACW00003k7', u'ACW00003qt', u'ACW000134w', u'ACW00004xa', u'ACW00015tr', u'ACW0000lkv', u'ACW0000kdn', u'ACW00007pw', u'ACW0000khw', u'ACW0000e4c', u'ACW00001j9', u'ACW00011yg', u'ACW00010wr', u'ACW0000jyd', u'ACW0000ldi', u'ACW00009bw', u'ACW0000e4v', u'ACW000173n', u'ACW0000w4h', u'ACW00017au', u'ACW00014gz', u'ACW00013qc', u'ACW000058b', u'ACW00006ld', u'ACW00011ry', u'ACW00012h1', u'ACW0000k9g', u'ACW00003m1', u'ACW0000tcl', u'ACW0000owj', u'ACW0000ody', u'ACW00016n1', u'ACW0000kbj', u'ACW00010lx', u'ACW00003ku', u'ACW0000wf2', u'ACW0000lle', u'ACW0000prb', u'ACW0000wxm', u'ACW0000ns2', u'ACW0000uez', u'ACW0000buz', u'ACW00003f4', u'ACW0000731', u'ACW000098h', u'ACW0000x0h', u'ACW0000qa1', u'ACW00012sd', u'ACW0002p6i', u'ACW0000ood', u'ACW00008sw', u'ACW0000bun', u'ACW0000qub', u'ACW0000lyk', u'ACW0000dd7', u'ACW00006j4', u'ACW0000xby', u'ACW000051f', u'ACW0000311', u'ACW0000t32', u'ACW0000y58', u'ACW0000qwy', u'ACW00007n1', u'ACW0000wo5', u'ACW0000bzq', u'ACW0000mq0', u'ACW0000rta', u'ACW00007bn', u'ACW0000fkw', u'ACW00008u1', u'ACW0000zcn', u'ACW0000kfb', u'ACW0000rms', u'ACW0000vlv', u'ACW00013fs', u'ACW0000xwr', u'ACW0000oej', u'ACW000112a', u'ACW0000jbl', u'ACW0000cu3', u'ACW00015w1', u'ACW0000wlj', u'ACW0000fjs', u'ACW0000hzn', u'ACW0000qx2', u'ACW00009do', u'ACW0000n80', u'ACW0000jro', u'ACW0000lzp', u'ACW0000paj', u'ACW0000sr7', u'ACW00008jn', u'ACW000124d', u'ACW0000tj2', u'ACW0000ioc', u'ACW0000lxd', u'ACW00010j9', u'ACW0000793', u'ACW0000ozm', u'ACW00013or', u'ACW00017ns', u'ACW00003iy', u'ACW0000snx', u'ACW0000mjd', u'ACW00017xf', u'ACW00011n5', u'ACW00007ny', u'ACW0000y5t', u'ACW00006nq', u'ACW00013fx', u'ACW00017gb', u'ACW0000lr0', u'ACW0000jzl', u'ACW000163c', u'ACW0000qh3', u'ACW0000g94', u'ACW0000yht', u'ACW0000ziv', u'ACW0005lik', u'ACW00012io', u'ACW00014r5', u'ACW00007ml', u'ACW0001r2v', u'ACW0000nd6', u'ACW000179h', u'ACW0000akh', u'ACW000107p', u'ACW0000ekq', u'ACW0000x28', u'ACW00013m1', u'ACW0000xyd', u'ACW0000oar', u'ACW00006ph', u'ACW0000wm3', u'ACW0000akv', u'ACW0000xqt', u'ACW0000fec', u'ACW0000u1o', u'ACW00014xt', u'ACW0000r5i', u'ACW00016ss', u'ACW0000ui4', u'ACW000187x', u'ACW0000tfi', u'ACW000070z', u'ACW00004u9', u'ACW0000xir', u'ACW00001t1', u'ACW000166p', u'ACW00011ds', u'ACW0000we9', u'ACW00007ex', u'ACW00012z8', u'ACW0001399', u'ACW000126v', u'ACW000055u', u'ACW0000ui1', u'ACW00013mb', u'ACW0000scr', u'ACW0000emy', u'ACW0000qad', u'ACW0000ki9', u'ACW00017l4', u'ACW00008ms', u'ACW00008xt', u'ACW0000byf', u'ACW0001nz2', u'ACW000126k', u'ACW0000je0', u'ACW0000xan', u'ACW0000dyc', u'ACW000039w', u'ACW0000lxn', u'ACW0000wec', u'ACW0000y07', u'ACW00002cu', u'ACW00016zk', u'ACW0000da8', u'ACW0000l56', u'ACW00014jj', u'ACW0000o4m', u'ACW0000rlj', u'ACW0000zbk', u'ACW0000hau', u'ACW0000tgh', u'ACW00003p2', u'ACW0001269', u'ACW0000ntw', u'ACW00011gs', u'ACW00013eo', u'ACW0000kcy', u'ACW00014jx', u'ACW0000rlz', u'ACW0000ws7', u'ACW00006h8', u'ACW00011h6']
i = 0
condor_pts = {}
condors_per_user = {}
classification_count = {}

check1 = 5
condors_at_1 = {}
check2 = 10
condors_at_2 = {}

total = 0
maxDiff = {}

check = [{} for i in range(0,11)]

for r in collection.find({"$and" : [{"tutorial":False}, {"subjects": {"$ne": []}} ]}):
    #zooniverse_id = r["zooniverse_id"]
    user_ip = r["user_ip"]
    zooniverse_id =  r["subjects"][0]["zooniverse_id"]

    #if zooniverse_id in toSkip:
    #    continue
    if not(zooniverse_id in toProcess):
        continue

    if not(zooniverse_id in condor_pts):
        condor_pts[zooniverse_id] = set()
        #condor_user_id[zooniverse_id] = []
        classification_count[zooniverse_id] = 0
        condors_per_user[zooniverse_id] = []

    classification_count[zooniverse_id] += 1
    condor_count = 0
    if "marks" in r["annotations"][-1]:
        markings = r["annotations"][-1].values()[0]

        for marking_index in markings:
            marking = markings[marking_index]
            try:
                if marking["animal"] == "condor":
                    scale = 1.875
                    x = scale*float(marking["x"])
                    y = scale*float(marking["y"])
                    condor_pts[zooniverse_id].add(((x,y),user_ip))
                    #condor_user_id[zooniverse_id].append(user_ip)
                    condor_count += 1
            except KeyError:
                continue

    condors_per_user[zooniverse_id].append(condor_count)

    #if (classification_count[zooniverse_id] == 5) and (condor_pts[zooniverse_id] != []):
    if classification_count[zooniverse_id] in range(3,11):
        #print zooniverse_id
        #print condor_pts[zooniverse_id]
        object_id = str(r["subjects"][0]["id"])
        url = r["subjects"][0]["location"]["standard"]
        if condor_pts[zooniverse_id] != set():
            try:
                xyPts,user_ids = zip(*list(condor_pts[zooniverse_id]))
                cluster_center =  adaptiveDBSCAN.adaptiveDBSCAN(xyPts,user_ids)# condor_pts[zooniverse_id],condor_user_id[zooniverse_id])
            except adaptiveDBSCAN.CannotSplit:
                print "bad image: " + r["subjects"][0]["location"]["standard"]
                continue

            num_condors = len(cluster_center)
        else:
            num_condors = 0


        check[classification_count[zooniverse_id]][zooniverse_id] = num_condors

        print len(check[10])
        if len(check[10]) == 300:
            break

    #for marking_index in r["annotations"][1]["value"]:

print check[10].keys()
x1 = [[] for i in range(0,11)]
for subject_id in check[10]:
    for i in range(4,11):
        x1[i].append(check[i][subject_id] - check[i-1][subject_id])

plt.plot(range(4,11), [np.mean(x1[i]) for i in range(4,11)])
plt.show()

# inBoth = [subject_id for subject_id in condors_at_1 if (subject_id in condors_at_2)]
# # print len(inBoth)
# #x = [condors_at_1[subject_id] for subject_id in inBoth]
# #y = [condors_at_2[subject_id] for subject_id in inBoth]
# x = [maxDiff[subject_id] for subject_id in inBoth]
# x = []
# for subject_id in inBoth:
#     mi = min(condors_per_user[subject_id])
#     ma = max(condors_per_user[subject_id])
#     if mi == ma:
#         x.append(0)
#     else:
#         normalized = [(c-mi)/float(ma-mi) for c in condors_per_user[subject_id]]
#         x.append(np.std(normalized))
# y = [condors_at_2[subject_id] - condors_at_1[subject_id] for subject_id in inBoth]
# #print zip(inBoth,zip(x,y))
# #plt.plot((0,20),(0,20),'--')
# # #print x
# # #print y
# plt.plot(x,y,'.')
# plt.show()

# if not(os.path.isfile("/home/greg/Databases/condors/images/"+object_id+".JPG")):
#     urllib.urlretrieve (url, "/home/greg/Databases/condors/images/"+object_id+".JPG")
#
# image_file = cbook.get_sample_data("/home/greg/Databases/condors/images/"+object_id+".JPG")
# image = plt.imread(image_file)
#
# fig, ax = plt.subplots()
# im = ax.imshow(image)
# #plt.show()
# #
# if cluster_center != []:
#     x,y = zip(*cluster_center)
#     plt.plot(x,y,'.',color='blue')
#
# plt.show()