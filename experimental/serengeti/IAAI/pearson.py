#!/usr/bin/env python
__author__ = 'greg'
from nodes import setup
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


photos,users = setup()
for p in photos.values():
    p.__majorityVote__()

X = []
Y = []
for u in users.values():
    e,h = u.__getStats__()

    if (e != -1) and (h != -1):
        X.append(e*100)
        Y.append(h*100)



plt.plot(X,Y,'.',color="black")
plt.xlabel("Percentage of easy pictures correctly classified")
plt.ylabel("Percentage of hard pictures correctly classified")
print pearsonr(X,Y)
plt.show()
