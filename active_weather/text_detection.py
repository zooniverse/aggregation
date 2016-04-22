#!/usr/bin/env python
import glob
import active_weather
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

height = 40#int(np.median(heights))
width = 30#int(np.median(widths))

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

def scale_img(img):
    res = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    res = (255-res)/255.
    res = np.reshape(res,784)

    array = np.ndarray((1,784))
    array[0,:] = res
    return array

# print(mnist.test.images[0].shape)
# print(type(probabilities.eval(feed_dict={x: mnist.test.images[0]}, session=sess)))
examples = []
for i in range(3000):
    l = mnist.test.labels[i]

    if l[5] == 1:
        # t = np.reshape(mnist.test.images[i],(28,28))
        # plt.imshow(t)
        # plt.show()
        #
        # break
        examples.append(mnist.test.images[i])

from sklearn.decomposition import PCA
print(len(examples))
pca = PCA(n_components=50)
X_r = pca.fit(np.asarray(examples)).transform(np.asarray(examples))
print(sum(pca.explained_variance_ratio_))
print(X_r.shape)
avg_2 = np.median(X_r,axis=0)

inverse = pca.inverse_transform(avg_2)
inverse = np.reshape(inverse,(28,28))
plt.imshow(inverse)
plt.show()

import math
probabilities=y


text = []



for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:40]:
    fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0009.JPG"
    img = active_weather.__extract_region__(fname)
    id_ = fname.split("/")[-1][:-4]
    print(id_)

    # set a baseline for performance with otsu's binarization
    mask = active_weather.__create_mask__(img)
    horizontal_grid, vertical_grid = active_weather.__cell_boundaries__(img)

    pca_image, threshold, inverted = active_weather.__pca__(img, active_weather.__otsu_bin__)

    masked_image = active_weather.__mask_lines__(pca_image, mask)
    # plt.imshow(masked_image)
    # plt.show()

    im2, contours, hierarchy = cv2.findContours(masked_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt,h in zip(contours,hierarchy[0]):
        if h[0] == -1:
            continue
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])[0]
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])[0]
        topmost = tuple(cnt[cnt[:, :, 1].argmax()][0])[1]
        bottommost = tuple(cnt[cnt[:, :, 1].argmin()][0])[1]

        perimeter = cv2.arcLength(cnt, True)


        # template2 = np.zeros(img.shape,np.uint8)
        # template2.fill(255)
        # cv2.drawContours(template2, [cnt], 0, 0, -1)
        # plt.imshow(template2)
        # plt.show()

        if (rightmost - leftmost > width) or (topmost - bottommost > height):
            continue

        if (rightmost - leftmost > 10) and (topmost - bottommost > 10) and (cv2.arcLength(cnt,True) > 10):
            # template = np.zeros((topmost - bottommost,rightmost - leftmost), np.uint8)
            # template = np.zeros((height,width), np.uint8)
            # template.fill(255)
            # print(template.shape)

            s = cnt.shape
            cnt = np.reshape(cnt,(s[0],s[2]))

            # cnt[:,0] -= leftmost
            # cnt[:,1] -= bottommost
            # cv2.drawContours(template, [cnt], 0, 0,-1)

            # print(leftmost,rightmost)
            template = masked_image[bottommost:topmost,leftmost:rightmost]
            # plt.imshow(masked_image[bottommost:topmost,leftmost:rightmost],cmap="gray")
            # plt.show()
            # print(template.shape)


            res = scale_img(template)
            T = pca.transform(np.asarray(res))
            print(np.power(np.sum(np.power(T-avg_2,2)),0.5))
            p = probabilities.eval(feed_dict={x: res}, session=sess)[0]

            res = np.reshape(res,(28,28))
            plt.imshow(res)
            print(np.max(p),np.argmax(p))
            plt.show()
    break