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

penguins_at,noise = pickle.load(open(base_directory+"/Databases/penguins_at_3.pickle","rb"))

from scipy.stats.stats import pearsonr

diff = [v20-v5 for v5,v20 in zip(penguins_at[10],penguins_at[20])]
print pearsonr(noise[5],diff)
plt.plot(noise[5],diff,'.')
plt.show()

#n, bins, patches = plt.hist(penguins_at[20], 10, normed=1,histtype='step', cumulative=True)
#plt.show()
