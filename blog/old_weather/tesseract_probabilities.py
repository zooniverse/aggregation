__author__ = 'ggdhines'
from mnist import MNIST
import tesseract

n_neighbors = 15

mndata = MNIST('/home/ggdhines/Databases/mnist')

testing = mndata.load_testing()

for a in