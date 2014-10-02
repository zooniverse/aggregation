#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo

val_5 = []
val_10 = []
val_15 = []
complete_list = []

client = pymongo.MongoClient()
db = client['penguin_2014-09-30']
collection = db["penguin_classifications"]

with open("/home/greg/Databases/penguinOut") as f:
    while True:
        header = f.readline()
        if not header: break

        zooniverse_id = f.readline()[:-1]
        #print zooniverse_id
        try:
            line_5 = f.readline()[:-1]
            v5 = int(line_5.split("-")[1])
            line_10 = f.readline()[:-1]
            v10 = int(line_10.split("-")[1])
            line_15 = f.readline()[:-1]
            v15 = int(line_15.split("-")[1])
            if v15 < 30:
                continue
            val_5.append(int(line_5.split("-")[1]))
            val_10.append(int(line_10.split("-")[1]))
            val_15.append(v15)
        except IndexError:
            continue

        complete = []

        for classification in collection.find({"subjects": {"$elemMatch": {"zooniverse_id": zooniverse_id}}}):
            try:
                #print len(classification["annotations"][1]["value"])
                if classification["annotations"][2]["value"] == "complete":
                    complete.append(1)
                else:
                    complete.append(0)
            except KeyError:
                #print classification["annotations"]
                pass

        #print complete
        if (np.mean(complete) > 0.4) and (v15 > 0):
            print zooniverse_id + " :: " + str(np.mean(complete)) + " - " +str(v10/float(v15)) + " && " + str(v15)
        complete_list.append(np.mean(complete))

        assert(len(val_15) == len(complete_list))

#plt.plot(val_5,val_10,'.')
#plt.plot(val_10,val_15,'.',color="green")
#plt.plot((0,100),(0,100))
r1 = [v5/float(v10) for v5,v10 in zip(val_5,val_10) if v10 > 0]
r2 = [v10/float(v15) for v10,v15 in zip(val_10,val_15)  if v15 > 0]
r2 = [v10/float(v15)  if v15>0 else -1 for v10,v15 in zip(val_10,val_15)]
d1 = [v10-v5 for v5,v10 in zip(val_5,val_10)]
d2 = [v15-v10 for v10,v15 in zip(val_10,val_15)]
#n, bins, patches = plt.hist(r2, 10, facecolor='green',cumulative=True,normed=1)
print len(complete_list)
print len(r2)
plt.plot(complete_list,r2,'.')
#plt.plot(d1,d2,'.')
plt.xlim((0,1))
plt.ylim((0,1))
plt.show()
