import glob
import active_weather
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

examples = []
labels = []
s_total = 0
e_total = 0

example_s = []
example_e = []

def scale_img(img):
    res = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    res = (255-res)/255.
    res = np.reshape(res,784)

    array = np.ndarray((1,784))
    array[0,:] = res
    return array

height_ = 35#int(np.median(heights))
width_ = 29#int(np.median(widths))

raw = []

for fname in glob.glob("/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/*.JPG")[:2]:
    # fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0009.JPG"

    img = active_weather.__extract_region__(fname)
    id_ = fname.split("/")[-1][:-4]
    print(id_)

    # set a baseline for performance with otsu's binarization
    mask = active_weather.__create_mask__(img)
    # horizontal_grid, vertical_grid = active_weather.__cell_boundaries__(img)

    pca_image, threshold, inverted = active_weather.__pca__(img, active_weather.__otsu_bin__)

    masked_image = active_weather.__mask_lines__(pca_image, mask)
    height = masked_image.shape[0]
    for text, conf, top, left, right, bottom in active_weather.__ocr_image__(masked_image):
        if conf < 85:
            continue
        if text == "S":
            character = masked_image[top:bottom,left:right]
            raw.append(character)
            character = cv2.resize(character, (width_, height_), interpolation=cv2.INTER_CUBIC)
            # plt.imshow(character)
            # plt.show()
            character = character.reshape(width_*height_)
            # print(character.shape,width,height)
            labels.append([1,0])
            example_s.append(character)
            s_total += 1
        elif text == "E":
            character = masked_image[top:bottom, left:right]
            labels.append([0,1])
            e_total += 1
            example_e.append(character)
        else:
            continue

        # examples.append(scale_img(character))


    # break

print(s_total,e_total)

from sklearn.decomposition import PCA
X = np.array(example_s)
print(X.shape)
pca = PCA(n_components=1)
pca.fit(X)
X_r = pca.transform(X)
print(X_r)

# for x,c in zip(X_r.tolist(),raw):
#     plt.imshow(c,cmap="gray")
#     plt.title(str(x))
#     plt.show()
#
#
# print(X_r)

# avg_x = np.median(X_r,axis=0)
# print(avg_x.shape)
# new_X = pca.inverse_transform(avg_x)
# new_X = new_X.reshape((height_,width_))
# print(sum(pca.explained_variance_ratio_))
# print(pca.explained_variance_ratio_)
# plt.imshow(new_X)
# plt.show()
# print(X_r.shape)

# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# y_ = tf.placeholder(tf.float32, [None, 10])
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y_i,y_value in enumerate(xrange(0, image.shape[0], stepSize)):
        for x_i,x_value in enumerate(xrange(0, image.shape[1], stepSize)):
            # yield the current window
            yield x_i,y_i,image[y_value:y_value + windowSize[1], x_value:x_value + windowSize[0]]

fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0009.JPG"
fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/Bear-AG-29-1940-0625.JPG"
img = active_weather.__extract_region__(fname)
id_ = fname.split("/")[-1][:-4]
print(id_)

# set a baseline for performance with otsu's binarization
mask = active_weather.__create_mask__(img)
horizontal_grid, vertical_grid = active_weather.__cell_boundaries__(img)

pca_image, threshold, inverted = active_weather.__pca__(img, active_weather.__otsu_bin__)

masked_image = active_weather.__mask_lines__(pca_image, mask)
step = 1
grid = np.zeros((masked_image.shape[0]/step+1,masked_image.shape[1]/step+1))
examples = []

for i,j,window in sliding_window(masked_image,step,(width_,height_)):
    try:
        res = np.reshape(window,width_*height_)
        # res = scale_img(window)
        # print(i, j)
        # print(res.shape)
        # assert False
        y = math.fabs(pca.transform(res))# - avg_x)
        # print(y,pca.transform(res) - avg_x)
        # grid[j, i] = y
        if y > 900:
            grid[j,i] = y

        examples.append((y,window))

        # if y < 300:
        #     plt.imshow(window,cmap="gray")
        #     plt.show()
    except ValueError,IndexError:
        pass

print(grid.shape)
plt.imshow(grid)
plt.show()

examples.sort(key = lambda x:x[0],reverse=True)
for a,x in examples:
    plt.imshow(x,cmap="gray")
    plt.title(str(a))
    plt.show()