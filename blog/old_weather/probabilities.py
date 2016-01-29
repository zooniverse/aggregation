__author__ = 'ggdhines'
from mnist import MNIST
from sklearn import neighbors
import numpy as np
from sklearn.neighbors import KernelDensity
import scipy
from scipy import interpolate
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n_neighbors = 15

mndata = MNIST('/home/ggdhines/Databases/mnist')
training = mndata.load_training()

weight = "distance"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)

pca = PCA(n_components=50)
T = pca.fit(training[0])
reduced_training = T.transform(training[0])
print sum(pca.explained_variance_ratio_)
# clf.fit(training[0], training[1])
clf.fit(reduced_training, training[1])

testing = mndata.load_testing()

X = []
Y = []

for test,ans in zip(testing[0],testing[1])[:2000]:
    r = T.transform(test)
    probabilities = list(clf.predict_proba(r)[0])
    max_prob = max(probabilities)
    most_likely_outcome = probabilities.index(max_prob)

    if most_likely_outcome == ans:
        Y.append(1)
    else:
        Y.append(0)

    X.append(max_prob)

b = 0.1

Y_2 = []

for x in np.arange(0.05,1.01,0.05):
    weights = [math.exp(-(x_0 - x)**2/(2*b**2)) for x_0 in X]
    new_y = sum([w*y for (w,y) in zip(weights,Y)])/float(sum(weights))

    Y_2.append(new_y)

plt.plot(np.arange(0.05,1.01,0.05),Y_2)
plt.savefig("/home/ggdhines/t.png")
