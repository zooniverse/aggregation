__author__ = 'greg'
import divisiveDBSCAN
import multiprocessing
import numpy as np
import math


class Worker(multiprocessing.Process):
    def __init__(self,min_samples,task_queue,result_queue):
        multiprocessing.Process.__init__(self)
        self.d_DBSCAN = divisiveDBSCAN.DivisiveDBSCAN(min_samples)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                break

            m_,u_,e_ = next_task
            noise,final,to_split = self.d_DBSCAN.binary_search_DBSCAN(m_,u_,e_,return_users=True)
            self.task_queue.task_done()
            self.result_queue.put((noise,final,to_split))

class DivisiveDBSCAN():
    def __init__(self, min_samples):
        self.min_samples = min_samples
        self.starting_epsilon = math.sqrt(1000**2 + 750**2)

    def fit(self, markings,user_ids,jpeg_file=None,debug=False):
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()

        num_workers = 3
        noise_points = []
        workers = [Worker(self.min_samples,tasks,results) for i in range(num_workers)]
        for w in workers:
            w.start()

        #start by creating the initial "super" cluster
        end_clusters = []
        clusters_to_go = [(markings[:],user_ids[:],self.starting_epsilon),]
        tasks_to_go = 0

        while True:
            #give all the tasks possible and then wait for more
            if (clusters_to_go == []) and (tasks_to_go == 0):
                break

            while clusters_to_go != []:
                tasks_to_go += 1
                tasks.put(clusters_to_go.pop(0))

            #get the results back from one of the workers
            noise,final,to_split = results.get()
            noise_points.extend(noise)
            tasks_to_go += -1

            end_clusters.extend(final[:])
            clusters_to_go.extend(to_split[:])

        for i in range(num_workers):
            tasks.put(None)

        cluster_centers = []
        for cluster in end_clusters:
            pts,users = cluster
            x,y = zip(*pts)
            cluster_centers.append((np.mean(x),np.mean(y)))

        if debug:
            return cluster_centers, end_clusters,noise_points
        else:
            return cluster_centers