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

penguins_at = pickle.load(open(base_directory+"/Databases/penguins_at.pickle","rb"))




plt.plot(penguins_at[5],penguins_at[20],'.')
plt.plot(penguins_at[10],penguins_at[20],'.',color="green")
plt.plot(penguins_at[15],penguins_at[20],'.',color="red")
plt.plot((30,30),(0,250),'--',color="black")
plt.plot((0,250),(0,250),color="black")
plt.show()

n, bins, patches = plt.hist(penguins_at[20], 10, normed=1,histtype='step', cumulative=True)
plt.show()