#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os
import math

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

marking_count= pickle.load(open(base_directory+"/Databases/penguins_at_3__.pickle","rb"))
#print marking_count
X = [x for x in range(31) if x in marking_count]
Y = [np.mean(marking_count[i]) for i in X]
err = [np.std(marking_count[i])/math.sqrt(len(marking_count[i])) for i in X]
plt.errorbar(X,Y,yerr=err)
Y2 = [np.median(marking_count[i]) for i in X]
plt.plot(X,Y2)
plt.plot((0,30),(0,30))
plt.show()
