__author__ = 'ggdhines'
import glob, os
import cPickle as pickle
# pickle.dump((c,n),open("/home/ggdhines/Dropbox/nn_cases/"+str(data_c)+".pic","wb"))
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer

dir = "/home/ggdhines/Dropbox/nn_cases/"

width = 13


net = buildNetwork(width**2, 20, 11,hiddenclass=TanhLayer)
ds = SupervisedDataSet(width**2, 11)

os.chdir(dir)
for fname in glob.glob("*.pic"):
    template = [0 for i in range(width**2)]

    (c,n) = pickle.load(open(dir+fname,"rb"))
    X,Y = zip(*c)

    offset_x = (max(X)+min(X))/2 - 6
    offset_y = (max(Y)+min(Y))/2 - 6

    for x,y in c:
        x_,y_ = x-offset_x,y-offset_y
        # print x_,y_
        template[y_*width+x_] = 1

    try:
        n = int(n)
    except ValueError:
        n = 10

    ds.addSample(template, (n,))

    # for y in range(width-1,-1,-1):
    #     for x in range(width):
    #         print template[x*width+y],
    #     print
    #
    # print

trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()

print n
print net.activate(template)