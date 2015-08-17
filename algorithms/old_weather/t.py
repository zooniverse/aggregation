__author__ = 'greg'
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
import numpy as np
import cv2

img = cv2.imread("/home/greg/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")

with open('old_weather.json') as data_file:
    data = json.load(data_file)

scale_x = 1.455
scale_y = 1.45
x_offset = 20
y_offset = 2

image_file = cbook.get_sample_data("/home/greg/Dropbox/066e48f5-812c-4b5f-ab04-df6c35f50393.jpeg")
image = plt.imread(image_file)
# print type(image[0])
# assert False

base = [198, 188, 153]

for box in data["classifications"][0]["annotations"]:
    if "type" in box.keys():
        continue

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    im = axes.imshow(image)
    x = int((box["x"]-x_offset)/scale_x)
    y = int((box["y"]-y_offset)/scale_y)

    x2 = int((box["x"] + box["width"]-x_offset)/scale_x)
    y2 = int((box["y"] + box["height"]-y_offset)/scale_y)



    # if ("type" in box.keys()) or (y < -1200):
    #     continue



    plt.plot([x,x2,x2,x,x],[y,y,y2,y2,y])


    plt.show()
    m = []

    for y_i in range(y,y2+1):
    # for x_i in range(x,x2+1):
        m_t = []
        # for y_i in range(y,y2+1):
        for x_i in range(x,x2+1):

            print x,y
            print x_i,y_i
            print

            p = image[y_i][x_i]
            m_t.append(p)

        m.append(np.asarray(m_t))
    # print np.asarray(m)
    # assert False
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    im = axes.imshow(np.asarray(m))

            # print p
            # print type(p)

            # if math.sqrt(sum([(a-b)**2 for (a,b) in zip(base,p)])) > 150:
            #     plt.plot(x_i,-y_i,'o',color="black")

    plt.show()

    for y_i in range(y,y2+1):
    # for x_i in range(x,x2+1):
        # for y_i in range(y,y2+1):
        for x_i in range(x,x2+1):
            p = image[y_i][x_i]
            if math.sqrt(sum([(a-b)**2 for (a,b) in zip(base,p)])) > 50:
                plt.plot(x_i,-y_i,'o',color="black")

    plt.show()

# plt.show()