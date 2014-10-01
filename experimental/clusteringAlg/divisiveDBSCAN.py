#!/usr/bin/env python
__author__ = 'greg'
from sklearn.cluster import DBSCAN
import numpy as np
import math
import matplotlib.cbook as cbook
from PIL import Image
import matplotlib.pyplot as plt

def dist(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

class CannotSplit(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return ""
samples_needed = 3


class DivisiveDBSCAN:
    def __init__(self, min_samples):
        self.min_samples = min_samples
        self.starting_epsilon = math.sqrt(1000**2 + 750**2)

    def binary_search_DBSCAN(self,markings,user_ids,starting_epsilon,jpeg_file=None):
        if jpeg_file is not None:
            image_file = cbook.get_sample_data(jpeg_file)
            image = plt.imread(image_file)
            fig, ax = plt.subplots()
            im = ax.imshow(image)

            x,y = zip(*markings)
            plt.plot(x,y,'.',color='blue')

        #check to see if all of the points are from the same user - if so, they are noise
        #and we can skip the rest of the method
        if len(set(user_ids)) == 1:
            return markings,[],[]
        #given starting_epsilon (and min_samples), all of the markings should be in the same cluster
        x_ = np.array(markings)
        min_epsilon = 0.
        max_epsilon = starting_epsilon
        #print "===" + str(starting_epsilon)
        print self.min_samples
        while (max_epsilon-min_epsilon) >= 0.01:
            mid_epsilon = (max_epsilon+min_epsilon)/2.
            #print mid_epsilon
            db = DBSCAN(eps=mid_epsilon, min_samples=self.min_samples).fit(x_)
            labels = db.labels_
            unique_labels = set(labels)

            if len(unique_labels) > 1:
                min_epsilon = mid_epsilon
            else:
                max_epsilon = mid_epsilon
            continue

            #if all of the resulting clusters still need to be split
            all_split = True
            for k in unique_labels:
                if k == -1:
                    continue
                u_ = [ip for index,ip in enumerate(user_ids) if labels[index] == k]
                #if a cluster does not need to be split any further
                if len(set(u_)) == len(u_):
                    all_split = False
                    break

            if all_split:
                max_epsilon = mid_epsilon
            else:
                min_epsilon = mid_epsilon

        #this is the epsilon we are going to be using
        if min_epsilon == 0:
            assert unique_labels == set([-1])
            return markings,[],[]
        assert min_epsilon > 0
        db = DBSCAN(eps=min_epsilon, min_samples=self.min_samples).fit(x_)
        #extract any and all clusters that do not need to be further split
        labels = db.labels_
        unique_labels = set(labels)

        #if all of the resulting clusters still need to be split
        noise_markings = []
        final_clusters = []
        to_split_further_clusters = []

        for k in unique_labels:
            x_ = [ip for index,ip in enumerate(markings) if labels[index] == k]

            #find out which points are in this cluster
            #this usually will not happen - but CAN
            if k == -1:
                noise_markings.extend(x_)
            else:
                u_ = [ip for index,ip in enumerate(user_ids) if labels[index] == k]
                #if a cluster does not need to be split any further - add it to the final clusters
                if len(set(u_)) == len(u_):
                    if len(x_) < self.min_samples:
                        print k
                        print labels

                    assert(len(x_)>= self.min_samples)
                    final_clusters.append(x_)
                else:
                    to_split_further_clusters.append((x_,u_,min_epsilon))

                if (k == 0) and (jpeg_file is not None):
                    x,y = zip(*x_)
                    plt.plot(x,y,'.',color='green')

        if jpeg_file is not None:
            plt.show()
        return noise_markings,final_clusters,to_split_further_clusters

    def fit(self, markings,user_ids,jpeg_file=None):
        #start by creating the initial "super" cluster
        end_clusters = []
        clusters_to_go = [(markings[:],user_ids[:],self.starting_epsilon),]

        while True:
            #if we have run out of clusters to process, break (hopefully done :) )
            if clusters_to_go == []:
                break
            m_,u_,e_ = clusters_to_go.pop(0)

            noise,final,to_split = self.binary_search_DBSCAN(m_,u_,e_,jpeg_file)
            end_clusters.extend(final[:])
            clusters_to_go.extend(to_split[:])

            #break

        cluster_centers = []
        for cluster in end_clusters:
            x,y = zip(*cluster)
            cluster_centers.append((np.mean(x),np.mean(y)))

        return cluster_centers

