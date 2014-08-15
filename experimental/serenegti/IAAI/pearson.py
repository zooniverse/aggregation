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
        X.append(e)
        Y.append(h)



plt.plot(X,Y,'.')
print pearsonr(X,Y)
plt.show()
