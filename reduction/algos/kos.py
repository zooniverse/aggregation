import numpy as np
from math import copysign
from numpy.random import normal


class KOS:
    def __init__(self, iterations=100):
        self.T = int(iterations)

    def __call__(self, graph):
        self.init_user(graph)
        for n in range(self.T):
            print("Iteration: " + str(n + 1) + " of " + str(self.T))
            self.sigma_task(graph)
            self.sigma_worker(graph)

        return self.answer_map(graph)

    def sigma_task(self, graph):
        for t, _, _, d in graph.tasks():
            t.p = np.sum([e['answer'] * w.p for e, w in d])

    def sigma_worker(self, graph):
        for w, _, _, _, d in graph.workers():
            w.p = np.sum([e['answer'] * t.p for e, t, in d])

    def init_user(self, graph):
        dist = normal(size=len(graph._workers._nodes))
        for i, (worker, _, _, _, _) in enumerate(graph.workers()):
            worker.p = dist[i]

    def answer_map(self, graph):
        self.sigma_task(graph)
        return [(t.id, copysign(1, t.p)) for t, _, _, _ in graph.tasks()]
