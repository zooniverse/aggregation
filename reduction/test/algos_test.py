from reduction.graph import Graph
from reduction.algos.lpi import LPI
from reduction.algos.kos import KOS
from random import randint, sample, choice
import unittest


class AlgosTest(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()

        tasks = []
        gold_tasks = []

        for x in range(1000):
            t_id = 't' + str(x)
            self.graph.add_task(t_id)
            tasks.append(t_id)

        for x in range(50):
            gt_id = 'gt' + str(x)
            self.graph.add_gold_task(gt_id, choice([-1, 1]))
            gold_tasks.append(gt_id)

        for w in range(100):
            w_id = "w" + str(w)
            self.graph.add_worker(w_id)
            seen_tasks = randint(0, 500)
            seen_gold_tasks = randint(0, 50)
            for task in sample(tasks, seen_tasks):
                self.graph.add_answer(w_id, task, choice([-1, 1, 1]))
            for gold_task in sample(gold_tasks, seen_gold_tasks):
                self.graph.add_answer(w_id, gold_task, choice([-1, 1, 1]))

    def test_lpi(self):
        lpi = LPI(iterations=1)
        output = lpi(self.graph)
        print(output[1:20])
        self.assertEqual(len(output[0]), 2)

    def test_kos(self):
        kos = KOS(iterations=1)
        output = kos(self.graph)
        print(output[1:20])
        self.assertEqual(len(output[0]), 2)


if __name__ == '__main__':
    unittest.main()
