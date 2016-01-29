__author__ = 'ggdhines'
from sklearn.cluster import KMeans as sk_KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import six
from matplotlib import colors
import math

class KMeans:
    def __init__(self, min_samples):
        self.min_samples = min_samples

    def fit2(self, markings,user_ids,jpeg_file=None,debug=False):
        for n in range(1,len(markings)):
            kmeans = sk_KMeans(init='k-means++', n_clusters=n, n_init=10).fit(markings)

            labels = kmeans.labels_
            unique_labels = set(labels)
            #need to check if all clusters are either "clean" or noise
            clean = True
            for k in unique_labels:
                users = [ip for index,ip in enumerate(user_ids) if labels[index] == k]

                if len(users) < self.min_samples:
                    continue

                #we have found a "clean" - final - cluster
                if len(set(users)) != len(users):
                    clean = False
                    break

            if clean:
                break

        print n
        return None,None,None