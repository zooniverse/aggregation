#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

results = pickle.load(open(base_directory+"/Databases/penguins_at.pickle","rb"))

max_5_10 = {}
for x,y in zip(results[5],results[20]):
    if not(x in max_5_10):
        max_5_10[x] = y
    else:
        max_5_10[x] = max(max_5_10[x],y)

#plt.plot(max_5_10.keys(),max_5_10.values(),'.')
plt.plot(results[10],results[20],'.',color="green")
plt.plot((0,100),(0,100))
plt.show()