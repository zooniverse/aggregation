__author__ = 'ggdhines'
import gzip
import cPickle
import matplotlib.pyplot as plt
import math

f = gzip.open('/home/ggdhines/github/neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

index = 0

scale1 = 28
scale2 = 40

for x in range(28):
    for y in range(28):
        if training_data[0][index][x*28+y] > 0:
            plt.plot(y,-x,"o",color="blue")

print training_data[1][index]
plt.show()

m = (scale1-1)/float(scale2-1)

for x in range(scale2):
    for y in range(scale2):
        l_x = math.floor(x*m)
        u_x = math.ceil(x*m)

        l_y = math.floor(y*m)
        u_y = math.ceil(y*m)

        print training_data[0][index][l_x*28+l_y]
        print training_data[0][index][l_x*28+u_y]
        print training_data[0][index][u_x*28+u_y]
        print training_data[0][index][u_x*28+l_y]
        print

        #x*m,y*m