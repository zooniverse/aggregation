#!/usr/bin/env python
import numpy as np

def weight(beta):
    TP = 22
    TN = 18
    FN = 1
    FP = 15
    if (TP+beta*TN + FP+FN) == 0:
        return -1
    return (TP+beta*TN)/float(TP+beta*TN + FP+FN)

import matplotlib.pyplot as plt
betas = np.arange(0,1,0.1)
values = [weight(b) for b in betas]

plt.plot(betas,values)

plt.show()