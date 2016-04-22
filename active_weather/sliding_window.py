#!/usr/bin/env python
from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.cluster import DBSCAN
import numpy as np

def get_window_size():
    non_white_points = np.where(img[:, :500] != 255)
    non_white_points = np.asarray(zip(non_white_points[0], non_white_points[1]))
    print(non_white_points.shape)
    db = DBSCAN(eps=1, min_samples=5).fit(non_white_points)
    labels = db.labels_

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    print("here")
    heights = []
    widths = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue
        # print(k)

        class_member_mask = (labels == k)
        xy = non_white_points[class_member_mask]
        #
        min_y, min_x = np.min(xy, axis=0)
        max_y, max_x = np.max(xy, axis=0)
        if min(max_x - min_x, max_y - min_y) <= 1:
            continue

        heights.append(max_y - min_y)
        widths.append(max_x - min_x)

def scale_img(img):
    res = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    res = (255-res)/255.
    res = np.reshape(res,784)

    array = np.ndarray((1,784))
    array[0,:] = res
    return array

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y_i,y_value in enumerate(xrange(0, image.shape[0], stepSize)):
        for x_i,x_value in enumerate(xrange(0, image.shape[1], stepSize)):
            # yield the current window
            yield x_i,y_i,image[y_value:y_value + windowSize[1], x_value:x_value + windowSize[0]]

img = cv2.imread("/home/ggdhines/test2.jpg",0)


    # plt.plot(xy[:, 1], -xy[:, 0], '.', markerfacecolor=col, markeredgecolor='k')

height = 35#int(np.median(heights))
width = 29#int(np.median(widths))

# plt.show()
# assert False
# for wind in sliding_window(img,5,(50,50)):
#     plt.imshow(wind[2])
#     plt.show()



x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(accuracy)
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# print("accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# prediction=tf.argmax(y,1)
# print("predictions", prediction.eval(feed_dict={x: mnist.test.images}, session=sess))

print(mnist.test.images[0].shape)
# print(type(probabilities.eval(feed_dict={x: mnist.test.images[0]}, session=sess)))
#
# for i in range(20):
#     t = np.reshape(mnist.test.images[i],(28,28))
#     plt.imshow(t)
#     plt.show()

grid = np.zeros((img.shape[0]/5+1,img.shape[1]/5+1))

probabilities=y

for x_i,y_i,window in sliding_window(img,5,(width,height)):
    res = scale_img(window)

    # max_prob = max(probabilities.eval(feed_dict={x: res}, session=sess)[0])
    max_prob = probabilities.eval(feed_dict={x: res}, session=sess)[0][5]
    if max_prob > 0.8:
        print(probabilities.eval(feed_dict={x: res}, session=sess)[0])
        print(sum(probabilities.eval(feed_dict={x: res}, session=sess)[0]))

        t = np.reshape(res, (28, 28))
        plt.imshow(t)
        plt.show()

        grid[y_i,x_i] = max_prob
    # print(x_i,y_i)

print(grid.shape)
plt.imshow(grid)
plt.show()

assert False

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# for i in range(1000):
#   batch = mnist.train.next_batch(50)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1]})

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(500):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

probabilities=y_conv

for x_i,y_i,window in sliding_window(img,5,(width,height)):
    res = scale_img(window)

    # max_prob = max(probabilities.eval(feed_dict={x: res}, session=sess)[0])
    max_prob = probabilities.eval(feed_dict={x: res,keep_prob: 1.0})[0][9]
    # print(res)
    if max_prob > 0.8:

        t = np.reshape(res, (28, 28))
        plt.imshow(t)
        plt.title(str(max_prob))
        plt.show()

        grid[y_i,x_i] = max_prob
    # print(x_i,y_i)

print(grid.shape)
plt.imshow(grid)
plt.show()