from numpy.random import beta as beta_dist
from math import exp, log, copysign
from scipy.special import beta
from itertools import chain
import numpy as np


class LPI:
    def __init__(self, iterations=100, alpha=2, beta=1):
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.T = int(iterations)

    def __call__(self, graph):
        self.init_user(graph)
        for n in range(self.T):
            print("Iteration: " + str(n + 1) + " of " + str(self.T))
            self.sigma_task(graph)
            self.sigma_worker(graph)

        return self.answer_map(graph)

    def sigma_task(self, graph):
        for t, degree, _, d in graph.tasks():
            weighted_ans = [edge['answer'] * worker.p for edge, worker in d]
            t.p = np.sum(weighted_ans)

    def local_factor(self, cj, qj):
        return beta(self.alpha + cj, self.beta + qj - cj)

    def exp_x(self, delta):
        delta_set = [exp(edge['answer'] * task.p) for edge, task in delta]
        x = [1]
        for i in delta_set:
            x = [a + b * i for a, b in zip(chain([0], x), chain(x, [0]))]
        return x

    def sigma_worker_frac(self, exp_x, alpha_j, gamma_j):
        return np.sum([self.local_factor(k + alpha_j, gamma_j) * exp_x[k] for k
                       in range(gamma_j)])

    def sigma_worker(self, graph):
        for w, gamma_j, alpha_j, _, delta in graph.workers():
            exp_x = self.exp_x(delta)
            numerator = self.sigma_worker_frac(exp_x, alpha_j + 1, gamma_j)
            denominator = self.sigma_worker_frac(exp_x, alpha_j, gamma_j)
            w.p = log(numerator/denominator)

    def init_user(self, graph):
        dist = beta_dist(self.alpha, self.beta, len(graph._workers._nodes))
        for i, (worker, _, _, _, _) in enumerate(graph.workers()):
            worker.p = dist[i]

    def answer_map(self, graph):
        self.sigma_task(graph)
        return [(t.id, copysign(1, t.p)) for t, _, _, _ in graph.tasks()]
