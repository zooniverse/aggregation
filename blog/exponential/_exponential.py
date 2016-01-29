__author__ = 'ggdhines'
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import chi2
from scipy.stats import norm

p_range = np.arange(0.05,1,0.05)
results = {p:[] for p in p_range}
A = {p:[] for p in p_range}
C = {p:[] for p in p_range}

num_samples = 100

for j in range(100):
    # lambda_ = random.uniform(0.1,0.6)
    lambda_ = random.expovariate(10)

    # samples = [random.expovariate(lambda_) for i in range(num_samples)]
    samples = [norm.ppf(random.uniform(0.5,1),scale=lambda_) for i in range(num_samples) ]


    mean = np.mean(samples)
    ub = (2*num_samples*mean)/chi2.ppf(0.025, 2*num_samples)
    lb = (2*num_samples*mean)/chi2.ppf(1-0.025, 2*num_samples)

    # print lb,mean,ub
    # assert False

    for p in p_range:


        lb_percentile = -math.log(1-p)*lb
        ub_percentile = -math.log(1-p)*ub

        predicted_m = -math.log(1-p)*mean

        actual_p = sum([1 for s in samples if s <= predicted_m])/float(len(samples))
        results[p].append(actual_p)


        A[p].append(len([1. for s in samples if s <= lb_percentile])/float(num_samples))
        C[p].append(len([1. for s in samples if s <= ub_percentile])/float(num_samples))

# plt.plot([0.3],[0.25],"o")
plt.plot([0,1],[0,1],"--",color="black")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Predicted Percentile")
plt.ylabel("Actual Percentile")

# plt.show()

Y = []
for p in p_range:
    Y.append(np.mean(results[p]))

plt.plot(p_range,Y,color="blue")

Y2 = []
for p in p_range:
    Y2.append(np.mean(A[p]))
plt.plot(p_range,Y2,color="green")

Y3 = []
for p in p_range:
    Y3.append(np.mean(C[p]))
plt.plot(p_range,Y3,color="red")

plt.savefig("/home/ggdhines/non_exponential2.png",)
plt.show()