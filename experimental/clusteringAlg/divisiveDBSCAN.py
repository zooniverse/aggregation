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

    def __own_DBSCAN__(self,epsilon,markings,debug=False):
        visited = {m:False for m in markings}
        in_cluster = {m:False for m in markings}

        def r1(m,m2):
            if math.sqrt(math.pow(m[0]-m2[0],2)+math.pow(m[1]-m2[1],2)) <= epsilon+0.05:
                return True
            else:
                return False

        def r2(m,m2):
            if ((m[0]-m2[0])**2+(m[1]-m2[1])**2)**0.5 <= epsilon:
                return True
            else:
                return False


        def query_region__(m):
            return [m2 for m2 in markings if r1(m,m2) and not(in_cluster[m2])]

        def cluster__(m, nearby_m):
            cluster = set([m])
            in_cluster[m] = True
            for m2 in nearby_m:
                if not(in_cluster[m2]):
                    cluster.add(m2)
                    in_cluster[m2] = True
                    if not(visited[m2]):
                        visited[m2] = True
                        nearby_m2 = query_region__(m2)
                        if len(nearby_m2) >= self.min_samples:
                            nearby_m.extend([m3 for m3 in nearby_m2 if not(m3 in nearby_m)])

            return cluster

        clusters = []
        for m in markings:
            if not(visited[m]):
                visited[m] = True
                nearby_m = query_region__(m)
                if len(nearby_m) >= self.min_samples:
                    clusters.append(cluster__(m,nearby_m))

        for c in clusters:
            assert(len(c) >= self.min_samples)

        labels = []
        #if debug:
        #    print clusters
        for m in markings:
            found = False
            for cluster_index,c in enumerate(clusters):
                if m in c:
                    labels.append(cluster_index)
                    found = True
                    break

            if not(found):
                labels.append(-1)

        return labels





    def binary_search_DBSCAN(self,markings,user_ids,starting_epsilon,return_users=False,jpeg_file=None):
        if jpeg_file is not None:
            image_file = cbook.get_sample_data(jpeg_file)
            image = plt.imread(image_file)
            fig, ax = plt.subplots()
            im = ax.imshow(image)

            x,y = zip(*markings)
            plt.plot(x,y,'.',color='blue')

        #double check if we actually need to be splitting this cluster - if not, just return it
        if len(list(set(user_ids))) == len(user_ids):
            if return_users:
                return [],[(markings,user_ids)],[]
            else:
                return [],[markings,],[]

        #check to see if all of the points are from the same user - if so, they are noise
        #and we can skip the rest of the method
        if len(set(user_ids)) == 1:
            noise_markings = []
            for x in markings:
                noise_markings.append((x,user_ids[0]))
            return noise_markings,[],[]
        #given starting_epsilon (and min_samples), all of the markings should be in the same cluster
        markings_nparray = np.array(markings)
        min_epsilon = 0.
        max_epsilon = starting_epsilon
        #print "===" + str(starting_epsilon)
        #print self.min_samples
        while (max_epsilon-min_epsilon) >= 0.01:
            mid_epsilon = (max_epsilon+min_epsilon)/2.
            db = DBSCAN(eps=mid_epsilon, min_samples=self.min_samples).fit(markings_nparray)
            labels = db.labels_
            unique_labels = set(labels)
            #labels = self.__own_DBSCAN__(mid_epsilon,markings)
            #unique_labels = set(labels)
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
            #print "there there"
            #print markings
            #print user_ids
            x,y = zip(*markings)
            #plt.plot(x,y,'.')
            #plt.show()

            noise_markings = []
            for x in markings:
                noise_markings.append((x,user_ids[0]))
            #print "here here"
            return noise_markings,[],[]

            #return markings,[],[]


        assert min_epsilon > 0
        #new_epsilon = int((min_epsilon*10))/10.
        db = DBSCAN(eps=min_epsilon, min_samples=self.min_samples).fit(markings_nparray)
        labels = db.labels_
        #extract any and all clusters that do not need to be further split
        #labels = self.__own_DBSCAN__(new_epsilon,markings)
        unique_labels = set(labels)


        #if all of the resulting clusters still need to be split
        noise_markings = []
        final_clusters = []
        to_split_further_clusters = []

        colors = ["green","blue","tan","lightseagreen"]

        for ii,k in enumerate(unique_labels):
            x_ = [ip for index,ip in enumerate(markings) if labels[index] == k]
            u_ = [ip for index,ip in enumerate(user_ids) if labels[index] == k]
            #find out which points are in this cluster
            #this usually will not happen - but CAN
            if k == -1:
                for x,u in zip(x_,u_):
                    assert(type(x) == tuple)
                    noise_markings.append((x,u))
                if (jpeg_file is not None):
                    x,y = zip(*x_)
                    plt.plot(x,y,'.',color="red")
            else:

                #if a cluster does not need to be split any further - add it to the final clusters
                if len(set(u_)) == len(u_):
                    if len(x_) < self.min_samples:
                        for x,u in zip(x_,u_):
                            assert(type(x) == tuple)
                            noise_markings.append((x,u))
                    else:
                        if return_users:
                            final_clusters.append((x_,u_))
                        else:
                            final_clusters.append(x_)
                else:
                    if len(x_) < self.min_samples:
                        for x,u in zip(x_,u_):
                            assert(type(x) == tuple)
                            noise_markings.append((x,u))
                    else:
                        to_split_further_clusters.append((x_,u_,min_epsilon))

                if (jpeg_file is not None):
                    x,y = zip(*x_)
                    plt.plot(x,y,'o',color=colors[ii])

        if jpeg_file is not None:
            plt.xlim(0,1000)
            plt.ylim(748,0)
            plt.show()
        #print "//" + str(noise_markings)
        return noise_markings,final_clusters,to_split_further_clusters

    def fit(self, markings,user_ids,jpeg_file=None,debug=False):
        #start by creating the initial "super" cluster
        end_clusters = []
        clusters_to_go = [(markings[:],user_ids[:],self.starting_epsilon),]
        total_noise = {}
        total_noise2 = []

        if jpeg_file is not None:
            image_file = cbook.get_sample_data(jpeg_file)
            image = plt.imread(image_file)
            fig, ax = plt.subplots()
            im = ax.imshow(image)

            x,y = zip(*markings)
            plt.plot(x,y,'o',color="green")
            plt.xlim(0,1000)
            plt.ylim(748,0)
            plt.show()

        while True:
            #if we have run out of clusters to process, break (hopefully done :) )
            if clusters_to_go == []:
                break
            m_,u_,e_ = clusters_to_go.pop(0)

            noise_found,final,to_split = self.binary_search_DBSCAN(m_,u_,e_,jpeg_file)
            #print to_split
            #print e_
            #print noise
            #print "=== " + str(noise_found)
            if noise_found != []:
                #print "==="

                for p,u in noise_found:
                    total_noise2.append(p)
                    assert(type(p) == tuple)
                    if not(u in total_noise):
                        total_noise[u] = [p]
                    else:
                        total_noise[u].append(p)
            #total_noise.extend(noise)
            end_clusters.extend(final[:])
            clusters_to_go.extend(to_split[:])

            #break

        cluster_centers = []
        for cluster in end_clusters:
            x,y = zip(*cluster)
            cluster_centers.append((np.mean(x),np.mean(y)))
        #print total_noise
        #print "===="

        if debug:
            return cluster_centers, end_clusters,total_noise2
        else:
            return cluster_centers

