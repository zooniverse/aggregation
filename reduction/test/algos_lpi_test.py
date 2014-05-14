from reduction.algos.lpi import LPI
from reduction.graph import Graph
from random import randint, sample, choice
import unittest

class TestLPI(unittest.TestCase):
    def setUp(self):
        self.lpi = LPI()
        self.graph = Graph()
        tasks = []
        gold_tasks = []

        for x in range(100000):
            t_id = 't' + str(x)
            self.graph.add_task(t_id)
            tasks.append(t_id)

        for x in range(5000):
            gt_id = 'gt' + str(x)
            self.graph.add_gold_task(gt_id)
            gold_tasks.append(gt_id)

        for w in range(10000):
            w_id = "w", str(x)
            self.graph.add_worker(w_id)
            seen_tasks = randint(0,100000)
            seen_gold_tasks = randint(0,5000)
            for task in sample(tasks, seen_tasks):
                self.graph.add_answer(w_id, task, choice([-1, 1]))
            for gold_task in sample(gold_tasks, seen_gold_tasks):
                self.graph.add_answer(w_id, gold_task, choice([-1, 1]))

    def test_function(self):
       self.assertEqual(type(self.lpi(self.graph)), List) 

