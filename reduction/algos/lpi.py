from numpy.random import beta as beta_dist
from numpy.polynomial.polynomial import Polynomial, polymul
from math import exp,log,copysign
import numpy as np
from scipy.special import beta
import itertools

class LPI:
    def __init__(self, iterations=100, alpha=2, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.T = iterations

    def __call__(self, graph):
        self.init_user(graph)
        for _ in itertools.repeat(None, self.T):
            self.sigma_task(graph)
            self.sigma_worker(graph)

        return self.answer_map(graph)

    def sigma_task(self, graph):
        for task in graph.tasks():
            weighted_ans = [edge.ans * worker.p for edge,worker in task.delta]
            task.p = np.sum(weighted_ans)

    def local_factor(self, cj, qj):
        return beta(self.alpha + cj, self.beta + qj - cj)

    def polymultiply(self, memo, poly):
        return polymul(memo, poly)

    def exp_x(self, delta):
        polys = [[exp(task.p), 1] for edge,task in delta]
        return reduce(self.polymultiply, polys)

    def sigma_worker_frac(self, exp_x, alpha, beta, gamma_j):
        accum = 0.0
        for k in range(gamma_j):
            accum += (self.local_factor(k + alpha, beta) * exp_x[k])
        return accum

    def sigma_worker(self, graph):
        for w in graph.workers():
            exp_x = self.exp_x(w.delta)
            gamma_j = len(w.delta)
            alpha = w.alpha_j
            beta =  gamma_j + w.alpha_j
            numerator = self.sigma_worker_frac(exp_x, alpha + 1, beta, gamma_j)
            denominator = self.sigma_worker_frac(exp_x, alpha, beta, gamma_j)
            w.p = log(numerator/denominator)

    def init_user(self, graph):
        workers = graph.workers()
        dist = beta_dist(self.alpha, self.beta, len(workers))
        for i,worker in enumerate(workers):
            worker.p = dist[i] 

    def answer_map(self, graph):
        self.sigma_task(graph)
        return [(task.id, copysign(1, task.p)) for task in graph.tasks()]

