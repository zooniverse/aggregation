__author__ = 'ggdhines'
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

prediction = tf.argmax(y,1)

probabilities=y
prob = probabilities.eval(feed_dict={x: mnist.test.images}, session=sess)
predicted_labels = prediction.eval(feed_dict={x: mnist.test.images}, session=sess)

correct_labels = np.argmax(mnist.test._labels,axis=1)

true_positives = []
false_positives = []

# for i in range(len(correct_labels)):
#     corr = correct_labels[i]
#     pred = predicted_labels[i]
#     a = prob[i][pred]
#     if corr == pred:
#         true_positives.append(a)
#     else:
#         false_positives.append(a)
#
# alphas = true_positives[:]
# alphas.extend(false_positives)
# alphas.sort()
# X = []
# Y = []
# for a in alphas:
#     X.append(len([x for x in false_positives if x >= a])/float(len(false_positives)))
#     Y.append(len([y for y in true_positives if y >= a])/float(len(true_positives)))
#
# print len(false_positives)
# print len(true_positives)
# plt.plot(X,Y)
# plt.plot([0,1],[0,1],"--",color="green")
# plt.xlabel("False Positive Count")
# plt.ylabel("True Positive Count")
# plt.show()

from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

print X_digits

import gzip
import cPickle

from sklearn.decomposition import PCA

n_samples = len(X_digits)

f = gzip.open('/home/ggdhines/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train, y_train = train_set
X_test, y_test = test_set


pca = PCA(n_components=100)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
# print sum(pca.explained_variance_ratio_)
# assert False

knn = neighbors.KNeighborsClassifier()
trained_classifier = knn.fit(X_train, y_train)

prob = trained_classifier.predict_proba(X_test)

predicted = trained_classifier.predict(X_test)

for i in range(len(X_test)):
    corr = y_test[i]
    pred = predicted[i]
    a = prob[i][pred]
    if corr == pred:
        true_positives.append(a)
    else:
        false_positives.append(a)

alphas = true_positives[:]
alphas.extend(false_positives)
alphas.sort()
X = []
Y = []
for a in alphas:
    X.append(len([x for x in false_positives if x >= a])/float(len(false_positives)))
    Y.append(len([y for y in true_positives if y >= a])/float(len(true_positives)))

print len(false_positives)
print len(true_positives)
plt.plot(X,Y)
plt.plot([0,1],[0,1],"--",color="green")
plt.xlabel("False Positive Count")
plt.ylabel("True Positive Count")
plt.show()
