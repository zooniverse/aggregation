__author__ = 'greg'
import math
import numpy as np


def hypothesis(theta,X):
    z = sum([t*x for t,x in zip(theta,X)])
    return 1/(1. + math.exp(-z))


def cost(hypothesis_value,actual):
    if actual == 1:
        return -math.log(hypothesis_value)
    else:
        return -math.log(1.-hypothesis_value)


def cost_function(hypothesis_param,X,Y):
    cost_list = [cost(hypothesis(hypothesis_param,x),y) for x,y in zip(X,Y)]
    return np.mean(cost_list)

def partial_cost_function(theta,instances,Y,j):
    return np.mean([(hypothesis(theta,X) - y)*X[j] for X,y in zip(instances,Y)])